"""Cross-source ID harmonization to OptimusKG-canonical CURIEs.

Builds the three small mapping tables from ``ReferenceData`` (cached) and exposes
join helpers. Genes canonicalize to Ensembl, drugs to ChEMBL, diseases to the
OT-canonical id reached via UMLS. Built once and shared across loaders.
"""

from __future__ import annotations

from functools import cached_property

import polars as pl

from am_diag.loaders.graph.curie import to_biolink_curie
from am_diag.loaders.graph.reference_data import ReferenceData


class Harmonizer:
    """Map native source ids to OptimusKG-canonical ids.

    Attributes:
        None public. Construct from a ``ReferenceData`` instance and reuse across
        loaders so the mapping tables are built only once.
    """

    def __init__(self, reference: ReferenceData) -> None:
        """Store the reference layer used to build the cached mapping tables."""
        self._ref = reference

    @cached_property
    def ensembl_ncbi(self) -> pl.DataFrame:
        """``(ensembl_id, ncbi_id)`` from HGNC symbol joined to OT symbol."""
        target = self._ref.opentargets_target.select("id", "approved_symbol")
        return (
            target.join(
                self._ref.gene_names,
                left_on="approved_symbol",
                right_on="symbol",
                how="inner",
            )
            .select(pl.col("id").alias("ensembl_id"), pl.col("ncbi_id"))
            .unique()
        )

    @cached_property
    def chembl_drugbank(self) -> pl.DataFrame:
        """``(chembl_id, drugbank_id)`` from OT drug-molecule cross-references."""
        return (
            self._ref.opentargets_drug_molecule.explode("cross_references")
            .unnest("cross_references")
            .filter(pl.col("source") == "drugbank")
            .explode("ids")
            .select(
                pl.col("id").alias("chembl_id"),
                ("DRUGBANK:" + pl.col("ids")).alias("drugbank_id"),
            )
        )

    @cached_property
    def umls_disease(self) -> pl.DataFrame:
        """``(id, umls_id)`` from OT disease ``db_xrefs`` starting with ``UMLS:``."""
        return (
            self._ref.opentargets_disease.select(
                pl.col("id"),
                pl.col("metadata").struct.field("db_xrefs").alias("db_xrefs"),
            )
            .explode("db_xrefs")
            .filter(pl.col("db_xrefs").is_not_null())
            .filter(pl.col("db_xrefs").str.starts_with("UMLS"))
            .select(
                pl.col("id"),
                pl.col("db_xrefs").str.replace("UMLS:", "").alias("umls_id"),
            )
            .unique()
        )

    def to_ensembl(self, df: pl.DataFrame, ncbi_col: str) -> pl.DataFrame:
        """Inner-join ``df[ncbi_col]`` to Ensembl, adding an ``ensembl_id`` column."""
        return df.join(
            self.ensembl_ncbi, left_on=ncbi_col, right_on="ncbi_id", how="inner"
        )

    def to_chembl(self, df: pl.DataFrame, drugbank_col: str) -> pl.DataFrame:
        """Inner-join ``df[drugbank_col]`` to ChEMBL, adding a ``chembl_id`` column."""
        return df.join(
            self.chembl_drugbank,
            left_on=drugbank_col,
            right_on="drugbank_id",
            how="inner",
        )

    def disease_umls_to_canonical(
        self, df: pl.DataFrame, umls_col: str
    ) -> pl.DataFrame:
        """Inner-join a UMLS id column to the OT-canonical disease ``id``."""
        return df.join(
            self.umls_disease, left_on=umls_col, right_on="umls_id", how="inner"
        )

    @staticmethod
    def biolink(df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Rewrite an OBO column to Biolink ``PREFIX_`` form in place."""
        return df.with_columns(to_biolink_curie(pl.col(column)).alias(column))
