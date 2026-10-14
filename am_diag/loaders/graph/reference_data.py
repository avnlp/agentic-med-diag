"""Cached reads of the shared reference inputs (Open Targets, HGNC, ontologies).

The reference inputs feed both ID harmonization and node enrichment, and several
are read by multiple loaders, so they are parsed at most once per run. Open Targets
``target`` / ``drug_molecule`` carry their rich fields under a ``metadata`` struct;
those are flattened here so the ``Harmonizer`` and ``NodeEnricher`` see one shape.
``disease`` keeps ``name`` / ``description`` top-level (``db_xrefs`` stays nested),
matching OptimusKG's bronze layout.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import polars as pl

from am_diag.loaders.graph.formats import read_obograph_json


def _maybe_unnest(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Unnest a struct column into top-level columns when it is present."""
    if column in df.columns:
        return df.unnest(column)
    return df


class ReferenceData:
    """Lazily read and cache the cross-source reference frames.

    Attributes:
        reference_dir: Root holding ``opentargets/``, ``gene_names/``, ``ontology/``.
    """

    def __init__(self, reference_dir: str | Path) -> None:
        """Store the reference root directory for lazy, cached table reads."""
        self.reference_dir = Path(reference_dir)

    @cached_property
    def opentargets_target(self) -> pl.DataFrame:
        """OT target (gene metadata + ``approved_symbol``), metadata flattened."""
        raw = pl.read_parquet(
            self.reference_dir / "opentargets" / "target" / "*.parquet"
        )
        return _maybe_unnest(raw, "metadata")

    @cached_property
    def opentargets_drug_molecule(self) -> pl.DataFrame:
        """OT drug molecule (ChEMBL ids, cross-references), metadata flattened."""
        raw = pl.read_parquet(
            self.reference_dir / "opentargets" / "drug_molecule" / "*.parquet"
        )
        return _maybe_unnest(raw, "metadata")

    @cached_property
    def opentargets_disease(self) -> pl.DataFrame:
        """OT disease (``id``, ``name``, ``description``, nested ``metadata``)."""
        return pl.read_parquet(
            self.reference_dir / "opentargets" / "disease" / "*.parquet"
        )

    @cached_property
    def gene_names(self) -> pl.DataFrame:
        """HGNC gene_names TSV mapping ``symbol`` to ``ncbi_id``.

        Read as all-string columns (``infer_schema_length=0``) so identifier
        columns stay strings and join cleanly with the string ids loaders carry.
        """
        return pl.read_csv(
            self.reference_dir / "gene_names" / "gene_names.tsv",
            separator="\t",
            infer_schema_length=0,
        )

    def ontology(self, name: str) -> pl.DataFrame:
        """Read an OBO Graph JSON ontology by stem (``mondo``/``hp``/``uberon``)."""
        return read_obograph_json(self.reference_dir / "ontology" / f"{name}.json")
