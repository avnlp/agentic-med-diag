"""Released-OptimusKG loader: reads the published gold Parquet with polars.

OptimusKG is distributed on Harvard Dataverse (DOI ``10.7910/DVN/IYNGEV``) as
Apache Parquet. The gold layer comes in three interchangeable layouts, each
covering all 10 node types and 27 edge types:

* per-type files under ``nodes/`` and ``edges/`` (for example
  ``nodes/gene.parquet``) whose ``properties`` column is a native struct,
* consolidated ``nodes.parquet`` and ``edges.parquet`` whose ``properties`` column
  is a JSON string, and
* the same consolidated format restricted to the largest connected component
  (``largest_connected_component_nodes.parquet`` and
  ``largest_connected_component_edges.parquet``), selected with
  ``largest_connected_component=True``.

Nodes are ``(id, label, properties)``. Edges are
``(from, to, label, relation, undirected, properties)``: ``relation`` and
``undirected`` are top-level columns, and each side carries
``properties.sources`` as a ``{direct, indirect}`` struct of dataset names.

This loader maps the layouts onto the am_diag-native seam. Node ``id`` becomes
``curie``, the label abbreviation becomes a PascalCase ``label``, and the edge
``relation`` becomes ``type`` (the edge ``label`` is kept as the structural label).
The ``undirected`` flag is kept as a property. ``sources`` becomes a list of the
direct dataset names, with the indirect ones under ``indirect_sources``.

OptimusKG keeps one edge per ``(from, to, label)`` and records the winning
``relation``. Where sources disagree, ``properties.relation_assertions`` lists every
source statement. This loader adds a flat ``asserted_relations`` list of the
distinct relations so that conflicts, such as a drug-disease pair asserted as both
``INDICATION`` and ``CONTRAINDICATION``, can be filtered in a query. Safety-sensitive
use should read it instead of ``type`` alone. The per-drug base64 structure files
(``mol_file_base64``, ``mol_image_base64``) are not stored.

The user supplies the files themselves (no ``optimuskg`` client dependency and no
auto-download); point ``gold_dir`` at the directory holding them. Obtain the data
via the ``optimuskg`` PyPI client or directly from the Dataverse record.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import polars as pl

from am_diag.common.schema import OPTIMUSKG_SCHEMA
from am_diag.common.schema.optimuskg import ABBREV_TO_LABEL
from am_diag.loaders.graph.base import GraphLoader


# Per-drug base64 structure files add tens of megabytes and are unused by retrieval.
_DROPPED_NODE_PROPERTIES = ("mol_file_base64", "mol_image_base64")

_LCC_PREFIX = "largest_connected_component"


def _split_sources(df: pl.DataFrame) -> pl.DataFrame:
    """Replace a ``{direct, indirect}`` ``sources`` struct with two string lists."""
    if not isinstance(df.schema.get("sources"), pl.Struct):
        return df
    return df.with_columns(
        pl.col("sources").struct.field("direct").alias("sources"),
        pl.col("sources").struct.field("indirect").alias("indirect_sources"),
    )


def _flatten_entries(df: pl.DataFrame) -> pl.DataFrame:
    """Turn ``Gene`` object lists into the string lists the other node types use.

    ``Gene`` nodes store ``synonyms`` as ``{label, source}`` objects and ``xrefs``
    as ``{id, source}`` objects, while every other node type stores plain strings.
    Cross-reference ids carry no prefix, so they become ``source:id`` strings.
    """
    out = df
    if _is_struct_list(out.schema.get("synonyms")):
        out = out.with_columns(
            pl.col("synonyms").list.eval(pl.element().struct.field("label"))
        )
    if _is_struct_list(out.schema.get("xrefs")):
        out = out.with_columns(
            pl.col("xrefs").list.eval(
                pl.concat_str(
                    pl.element().struct.field("source"),
                    pl.element().struct.field("id"),
                    separator=":",
                )
            )
        )
    return out


def _is_struct_list(dtype: pl.DataType | None) -> bool:
    return isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct)


class OptimusKGGraphLoader(GraphLoader):
    """Load the released OptimusKG gold graph from a local Parquet directory."""

    dataset_name: ClassVar[str] = "optimuskg"
    graph_schema: ClassVar = OPTIMUSKG_SCHEMA

    def __init__(
        self, gold_dir: str | Path, *, largest_connected_component: bool = False
    ) -> None:
        """Initialize the loader.

        Args:
            gold_dir: Directory containing either ``nodes/`` and ``edges/``
                subdirectories of per-type Parquet files, or consolidated
                ``nodes.parquet`` and ``edges.parquet`` files.
            largest_connected_component: Read the
                ``largest_connected_component_nodes.parquet`` and
                ``largest_connected_component_edges.parquet`` files instead of the
                full graph. Both files must be present in ``gold_dir``.
        """
        self._gold = Path(gold_dir)
        self._lcc = largest_connected_component

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Read and map the gold Parquet into am_diag-native node/edge frames."""
        nodes = self._map_nodes(self._read_side("nodes"))
        edges = self._map_edges(self._read_side("edges"))
        return nodes, edges

    def _read_side(self, kind: str) -> pl.DataFrame:
        """Read one side ("nodes"/"edges"), flattening ``properties`` to columns."""
        if self._lcc:
            component = self._gold / f"{_LCC_PREFIX}_{kind}.parquet"
            if not component.is_file():
                raise FileNotFoundError(
                    f"Expected {component} for the largest connected component"
                )
            return self._flatten_json(pl.read_parquet(component))
        subdir = self._gold / kind
        if subdir.is_dir():
            files = sorted(subdir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No Parquet files in {subdir}")
            frames = [self._flatten_struct(pl.read_parquet(f)) for f in files]
            return pl.concat(frames, how="diagonal_relaxed")
        consolidated = self._gold / f"{kind}.parquet"
        if consolidated.is_file():
            return self._flatten_json(pl.read_parquet(consolidated))
        raise FileNotFoundError(
            f"Expected {subdir}/ or {consolidated} under {self._gold}"
        )

    @staticmethod
    def _flatten_struct(df: pl.DataFrame) -> pl.DataFrame:
        """Unnest a per-type file's native ``properties`` struct into columns."""
        if "properties" in df.columns:
            return _flatten_entries(df.unnest("properties"))
        return df

    @staticmethod
    def _flatten_json(df: pl.DataFrame) -> pl.DataFrame:
        """Decode a consolidated file's JSON ``properties`` string into columns.

        Each label is decoded on its own because the same key can hold different
        shapes for different labels, and a single inferred schema cannot cover both.
        """
        if "properties" not in df.columns:
            return df
        frames = []
        for part in df.partition_by("label", maintain_order=True):
            coalesced = part.with_columns(
                pl.when(pl.col("properties").str.len_bytes() == 0)
                .then(pl.lit("{}"))
                .otherwise(pl.col("properties"))
                .alias("properties")
            )
            # Series-level json_decode infers the union struct schema across all
            # rows; the expression form requires an explicit dtype in this polars
            # version.
            decoded = coalesced.get_column("properties").str.json_decode(
                infer_schema_length=None
            )
            frames.append(
                _flatten_entries(coalesced.with_columns(decoded).unnest("properties"))
            )
        return pl.concat(frames, how="diagonal_relaxed")

    @staticmethod
    def _map_nodes(df: pl.DataFrame) -> pl.DataFrame:
        """Rename ``id`` to ``curie``, map labels, and normalize ``sources``."""
        out = _split_sources(df.drop(_DROPPED_NODE_PROPERTIES, strict=False))
        out = out.rename({"id": "curie"}).with_columns(
            pl.col("label").replace(
                old=list(ABBREV_TO_LABEL.keys()),
                new=list(ABBREV_TO_LABEL.values()),
            )
        )
        if "name" not in out.columns:
            out = out.with_columns(pl.col("curie").alias("name"))
        return out

    @staticmethod
    def _map_edges(df: pl.DataFrame) -> pl.DataFrame:
        """Map ``relation`` to ``type`` and flatten ``sources`` and assertions."""
        out = _split_sources(df)
        if isinstance(out.schema.get("relation_assertions"), pl.List):
            out = out.with_columns(
                pl.col("relation_assertions")
                .list.eval(pl.element().struct.field("relation"))
                .list.unique()
                .list.sort()
                .alias("asserted_relations")
            )
        if "relation" in out.columns:
            out = out.rename({"relation": "type"})
        return out
