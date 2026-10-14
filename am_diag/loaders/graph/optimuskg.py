"""Released-OptimusKG loader: reads the published gold Parquet with polars.

OptimusKG is distributed on Harvard Dataverse (DOI ``10.7910/DVN/IYNGEV``) as
Apache Parquet. The gold layer comes in two interchangeable layouts:

* per-type files under ``nodes/`` and ``edges/`` (e.g. ``nodes/gene.parquet``)
  whose ``properties`` column is a native struct, and
* consolidated ``nodes.parquet`` / ``edges.parquet`` (and the ``largest_connected_
  component_*`` variants) whose ``properties`` column is a JSON string.

Nodes are ``(id, label, properties)``; edges are ``(from, to, label, properties)``
where the per-edge semantic ``relation``, ``undirected`` flag, and ``sources`` live
inside ``properties``. This loader maps both layouts onto the am_diag-native seam:
node ``id`` -> ``curie``, the ``input_label`` abbreviation -> a PascalCase
``label``, the edge ``relation`` -> ``type`` (the ``input_label`` is kept as the
structural ``label``), and the ``undirected`` flag is dropped (am_diag stores one
directed edge).

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


class OptimusKGGraphLoader(GraphLoader):
    """Load the released OptimusKG gold graph from a local Parquet directory."""

    dataset_name: ClassVar[str] = "optimuskg"
    graph_schema: ClassVar = OPTIMUSKG_SCHEMA

    def __init__(self, gold_dir: str | Path) -> None:
        """Initialize the loader.

        Args:
            gold_dir: Directory containing either ``nodes/`` and ``edges/``
                subdirectories of per-type Parquet files, or consolidated
                ``nodes.parquet`` and ``edges.parquet`` files.
        """
        self._gold = Path(gold_dir)

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Read and map the gold Parquet into am_diag-native node/edge frames."""
        nodes = self._map_nodes(self._read_side("nodes"))
        edges = self._map_edges(self._read_side("edges"))
        return nodes, edges

    def _read_side(self, kind: str) -> pl.DataFrame:
        """Read one side ("nodes"/"edges"), flattening ``properties`` to columns."""
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
            return df.unnest("properties")
        return df

    @staticmethod
    def _flatten_json(df: pl.DataFrame) -> pl.DataFrame:
        """Decode a consolidated file's JSON ``properties`` string into columns."""
        if "properties" not in df.columns:
            return df
        coalesced = df.with_columns(
            pl.when(pl.col("properties").str.len_bytes() == 0)
            .then(pl.lit("{}"))
            .otherwise(pl.col("properties"))
            .alias("properties")
        )
        # Series-level json_decode infers the union struct schema across all rows;
        # the expression form requires an explicit dtype in this polars version.
        decoded = coalesced.get_column("properties").str.json_decode(
            infer_schema_length=None
        )
        return coalesced.with_columns(decoded).unnest("properties")

    @staticmethod
    def _map_nodes(df: pl.DataFrame) -> pl.DataFrame:
        """Rename ``id`` to ``curie`` and map ``input_label`` to a PascalCase label."""
        out = df.rename({"id": "curie"}).with_columns(
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
        """Map ``relation``->``type``, keeping the ``undirected`` flag."""
        out = df
        if "relation" in out.columns:
            out = out.rename({"relation": "type"})
        return out
