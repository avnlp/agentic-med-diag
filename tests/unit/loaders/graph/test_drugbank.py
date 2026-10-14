"""Tests for the DrugBank drug-target / drug-drug loader."""

from __future__ import annotations

from pathlib import Path

from am_diag.loaders.graph.drugbank import DrugBankGraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData


_POLY_HEADER = "Species,UniProt ID,Drug IDs,Name\n"


def _write_drugbank(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "drugbank_gene_map.tsv").write_text(
        "UniProt ID(supplied by UniProt)\tNCBI Gene ID(supplied by NCBI)\nP1\t7157\n",
        encoding="utf-8",
    )
    (directory / "drugbank_all_target_polypeptide_ids.csv").write_text(
        _POLY_HEADER + "Humans,P1,DB00945,p53 protein\n", encoding="utf-8"
    )
    for role in ("carrier", "enzyme", "transporter"):
        (directory / f"drugbank_all_{role}_polypeptide_ids.csv").write_text(
            _POLY_HEADER, encoding="utf-8"
        )
    (directory / "full database.xml").write_text(
        '<?xml version="1.0"?>'
        '<drugbank xmlns="http://www.drugbank.ca">'
        "<drug><drugbank-id>DB00945</drugbank-id>"
        "<drug-interactions><drug-interaction>"
        "<drugbank-id>DB01050</drugbank-id><description>bleeding risk</description>"
        "</drug-interaction></drug-interactions></drug>"
        "<drug><drugbank-id>DB01050</drugbank-id></drug>"
        "</drugbank>",
        encoding="utf-8",
    )


class TestDrugBankLoader:
    """Drug-target role mapping, drug-drug interactions, and node harmonization."""

    def test_build_frames(self, reference_dir: Path) -> None:
        """Targets map to TARGET DRG-GEN edges; interactions to DRG-DRG edges."""
        src = reference_dir / "drugbank"
        _write_drugbank(src)
        reference = ReferenceData(reference_dir)
        loader = DrugBankGraphLoader(
            src,
            harmonizer=Harmonizer(reference),
            enricher=NodeEnricher(reference),
        )
        nodes, edges = loader.build_frames()

        drug_gene = edges.filter(edges["label"] == "DRG-GEN").row(0, named=True)
        assert drug_gene["from"] == "CHEMBL25"
        assert drug_gene["to"] == "ENSG1"
        assert drug_gene["type"] == "TARGET"

        drug_drug = edges.filter(edges["label"] == "DRG-DRG").row(0, named=True)
        assert drug_drug["type"] == "SYNERGISTIC_INTERACTION"
        assert {drug_drug["from"], drug_drug["to"]} == {"CHEMBL25", "CHEMBL100"}
        assert drug_drug["interaction_description"] == "bleeding risk"

        drug = nodes.filter(nodes["curie"] == "CHEMBL25").row(0, named=True)
        assert drug["label"] == "Drug"
        assert drug["name"] == "aspirin"

    def test_schema_covers_emitted_types(self, reference_dir: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        src = reference_dir / "drugbank"
        _write_drugbank(src)
        reference = ReferenceData(reference_dir)
        loader = DrugBankGraphLoader(
            src, harmonizer=Harmonizer(reference), enricher=NodeEnricher(reference)
        )
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
