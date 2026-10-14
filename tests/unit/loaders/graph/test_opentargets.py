"""Tests for the Open Targets multi-relation loader."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.opentargets import OpenTargetsGraphLoader
from am_diag.loaders.graph.reference_data import ReferenceData


def _write_ot(root: Path) -> None:
    ot = root / "opentargets"

    def _write(name: str, frame: pl.DataFrame) -> None:
        directory = ot / name
        directory.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(directory / "part-0.parquet")

    _write(
        "target_disease_associations",
        pl.DataFrame(
            {
                "disease_id": ["MONDO_0005148", "HP_0001", "GO_0008150"],
                "target_id": ["ENSG1", "ENSG2", "ENSG1"],
                "metadata": [
                    {"score": 0.9, "evidence_count": 3},
                    {"score": 0.5, "evidence_count": 1},
                    {"score": 0.7, "evidence_count": 2},
                ],
            }
        ),
    )
    _write(
        "drug_indication",
        pl.DataFrame(
            {
                "id": ["CHEMBL25"],
                "metadata": [
                    {
                        "indications": [
                            {
                                "disease": "MONDO_0005148",
                                "efo_name": "type 2 diabetes",
                                "max_phase_for_indication": 4.0,
                                "references": ["PMID:1"],
                            }
                        ],
                        "indication_count": 1,
                    }
                ],
            }
        ),
    )
    _write(
        "drug_mechanism_of_action",
        pl.DataFrame(
            {
                "targets": [["ENSG1"]],
                "chembl_ids": [["CHEMBL25"]],
                "metadata": [
                    {
                        "action_type": "INHIBITOR",
                        "target_name": "p53",
                        "target_type": "protein",
                        "references": ["PMID:2"],
                    }
                ],
            }
        ),
    )
    _write(
        "disease_phenotype",
        pl.DataFrame(
            {
                "disease": ["MONDO_0005148"],
                "phenotype": ["HP_0001"],
                "metadata": [{"evidence": ["x"]}],
            }
        ),
    )


class TestOpenTargetsLoader:
    """Associations split by namespace; drug, hierarchy, and phenotype edges build."""

    def test_build_frames(self, reference_dir: Path) -> None:
        """All Open Targets edge families build with correct labels/relations."""
        _write_ot(reference_dir)
        reference = ReferenceData(reference_dir)
        loader = OpenTargetsGraphLoader(
            reference_dir, reference=reference, enricher=NodeEnricher(reference)
        )
        nodes, edges = loader.build_frames()

        labels = {(r["from"], r["to"]): r for r in edges.iter_rows(named=True)}
        assert labels[("MONDO_0005148", "ENSG1")]["label"] == "DIS-GEN"
        assert labels[("MONDO_0005148", "ENSG1")]["type"] == "ASSOCIATED_WITH"
        assert labels[("HP_0001", "ENSG2")]["label"] == "PHE-GEN"
        assert labels[("GO_0008150", "ENSG1")]["label"] == "BPO-GEN"
        assert labels[("MONDO_0005015", "MONDO_0005148")]["type"] == "PARENT"
        assert labels[("MONDO_0005148", "HP_0001")]["type"] == "HAS_PHENOTYPE"
        assert labels[("CHEMBL25", "MONDO_0005148")]["type"] == "INDICATION"
        assert labels[("CHEMBL25", "ENSG1")]["type"] == "INHIBITOR"
        assert labels[("CHEMBL25", "ENSG1")]["label"] == "DRG-GEN"

        gene = nodes.filter(nodes["curie"] == "ENSG1").row(0, named=True)
        assert gene["label"] == "Gene"
        assert gene["symbol"] == "TP53"
        drug = nodes.filter(nodes["curie"] == "CHEMBL25").row(0, named=True)
        assert drug["label"] == "Drug"
        assert drug["name"] == "aspirin"

    def test_schema_covers_emitted_types(self, reference_dir: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        _write_ot(reference_dir)
        reference = ReferenceData(reference_dir)
        loader = OpenTargetsGraphLoader(
            reference_dir, reference=reference, enricher=NodeEnricher(reference)
        )
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
