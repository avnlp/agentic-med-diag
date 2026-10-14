"""Tests for NodeEnricher: per-entity-type metadata joins over reference data."""

from __future__ import annotations

import polars as pl

from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.reference_data import ReferenceData


class TestNodeEnricher:
    """Each enrich_* left-joins metadata and preserves unmatched rows as nulls."""

    def test_enrich_genes_adds_metadata(self, reference_data: ReferenceData) -> None:
        """Gene metadata is joined by Ensembl id; unmatched rows stay null."""
        nodes = pl.DataFrame({"curie": ["ENSG1", "ENSG_missing"]})
        out = NodeEnricher(reference_data).enrich_genes(nodes)
        matched = out.filter(pl.col("curie") == "ENSG1").row(0, named=True)
        assert matched["symbol"] == "TP53"
        assert matched["biotype"] == "protein_coding"
        missing = out.filter(pl.col("curie") == "ENSG_missing").row(0, named=True)
        assert missing["symbol"] is None

    def test_enrich_drugs_adds_metadata(self, reference_data: ReferenceData) -> None:
        """Drug metadata is joined by ChEMBL id."""
        nodes = pl.DataFrame({"curie": ["CHEMBL25"]})
        out = NodeEnricher(reference_data).enrich_drugs(nodes)
        assert out.row(0, named=True)["drug_name"] == "aspirin"

    def test_enrich_diseases_adds_metadata(self, reference_data: ReferenceData) -> None:
        """Disease metadata is joined by canonical id."""
        nodes = pl.DataFrame({"curie": ["MONDO_0005148"]})
        out = NodeEnricher(reference_data).enrich_diseases(nodes)
        assert out.row(0, named=True)["disease_name"] == "type 2 diabetes mellitus"

    def test_enrich_from_ontology_adds_terms(
        self, reference_data: ReferenceData
    ) -> None:
        """Ontology term metadata is joined by Biolink id."""
        nodes = pl.DataFrame({"curie": ["MONDO_0005148"]})
        out = NodeEnricher(reference_data).enrich_from_ontology(nodes, "mondo")
        row = out.row(0, named=True)
        assert row["term_name"] == "type 2 diabetes mellitus"
        assert row["synonyms"] == ["T2DM"]
