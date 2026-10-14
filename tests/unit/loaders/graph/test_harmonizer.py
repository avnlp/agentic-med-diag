"""Tests for the ID Harmonizer: mapping tables, join helpers, CURIE rewriting."""

from __future__ import annotations

import polars as pl

from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData


class TestHarmonizer:
    """Mapping-table construction and join helpers over synthetic reference data."""

    def test_ensembl_ncbi_maps_symbol_to_ids(
        self, reference_data: ReferenceData
    ) -> None:
        """The Ensembl-to-NCBI table is built by joining symbols across sources."""
        mapping = Harmonizer(reference_data).ensembl_ncbi.sort("ensembl_id")
        assert mapping.to_dicts() == [
            {"ensembl_id": "ENSG1", "ncbi_id": "7157"},
            {"ensembl_id": "ENSG2", "ncbi_id": "1956"},
        ]

    def test_chembl_drugbank_from_cross_references(
        self, reference_data: ReferenceData
    ) -> None:
        """The ChEMBL-to-DrugBank table is read from drug cross-references."""
        mapping = Harmonizer(reference_data).chembl_drugbank.sort("chembl_id")
        assert mapping.to_dicts() == [
            {"chembl_id": "CHEMBL100", "drugbank_id": "DRUGBANK:DB01050"},
            {"chembl_id": "CHEMBL25", "drugbank_id": "DRUGBANK:DB00945"},
        ]

    def test_umls_disease_filters_umls_xrefs(
        self, reference_data: ReferenceData
    ) -> None:
        """The UMLS-to-disease table keeps only UMLS cross-references."""
        mapping = Harmonizer(reference_data).umls_disease
        assert mapping.to_dicts() == [{"id": "MONDO_0005148", "umls_id": "C0011860"}]

    def test_to_ensembl_inner_joins(self, reference_data: ReferenceData) -> None:
        """to_ensembl inner-joins, dropping rows without an Ensembl mapping."""
        df = pl.DataFrame({"ncbi_gene_id": ["7157", "999"]})
        out = Harmonizer(reference_data).to_ensembl(df, "ncbi_gene_id")
        assert out.to_dicts() == [{"ncbi_gene_id": "7157", "ensembl_id": "ENSG1"}]

    def test_disease_umls_to_canonical(self, reference_data: ReferenceData) -> None:
        """disease_umls_to_canonical resolves UMLS ids to the canonical disease id."""
        df = pl.DataFrame({"umls_id": ["C0011860"]})
        out = Harmonizer(reference_data).disease_umls_to_canonical(df, "umls_id")
        assert out.to_dicts() == [{"umls_id": "C0011860", "id": "MONDO_0005148"}]

    def test_biolink_rewrites_obo_colon(self, reference_data: ReferenceData) -> None:
        """Biolink rewrites OBO colon ids to underscore form, leaving others intact."""
        df = pl.DataFrame({"curie": ["UBERON:0002107", "ENSG1"]})
        out = Harmonizer.biolink(df, "curie")
        assert out["curie"].to_list() == ["UBERON_0002107", "ENSG1"]
