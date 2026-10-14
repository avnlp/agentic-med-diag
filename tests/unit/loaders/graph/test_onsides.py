"""Tests for the OnSIDES adverse-drug-reaction loader."""

from __future__ import annotations

import zipfile
from pathlib import Path

from am_diag.loaders.graph.onsides import OnsidesGraphLoader


def _write_onsides(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(directory / "onsides.zip", "w") as archive:
        archive.writestr(
            "csv/high_confidence.csv",
            "ingredient_id,effect_meddra_id\n161,10028813\n",
        )
        archive.writestr(
            "csv/vocab_rxnorm_ingredient.csv",
            "rxnorm_id,rxnorm_name,rxnorm_term_type\n161,aspirin,IN\n",
        )
        archive.writestr(
            "csv/vocab_meddra_adverse_effect.csv",
            "meddra_id,meddra_name,meddra_term_type\n10028813,nausea,PT\n",
        )


class TestOnsidesLoader:
    """Drug-effect pairs become DRG-PHE edges with native RxNorm/MedDRA ids."""

    def test_build_frames(self, tmp_path: Path) -> None:
        """High-confidence pairs map to ADVERSE_DRUG_REACTION edges with names."""
        _write_onsides(tmp_path)
        nodes, edges = OnsidesGraphLoader(tmp_path).build_frames()

        assert edges.height == 1
        edge = edges.row(0, named=True)
        assert edge["from"] == "RXNORM:161"
        assert edge["to"] == "meddra:10028813"
        assert edge["type"] == "ADVERSE_DRUG_REACTION"
        assert edge["label"] == "DRG-PHE"

        drug = nodes.filter(nodes["curie"] == "RXNORM:161").row(0, named=True)
        assert drug["label"] == "Drug"
        assert drug["name"] == "aspirin"
        pheno = nodes.filter(nodes["curie"] == "meddra:10028813").row(0, named=True)
        assert pheno["label"] == "Phenotype"
        assert pheno["name"] == "nausea"

    def test_schema_covers_emitted_types(self, tmp_path: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        _write_onsides(tmp_path)
        loader = OnsidesGraphLoader(tmp_path)
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
