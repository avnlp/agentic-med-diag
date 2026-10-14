"""Tests for the PPI gene-gene interaction loader."""

from __future__ import annotations

from pathlib import Path

from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.ppi import PpiGraphLoader
from am_diag.loaders.graph.reference_data import ReferenceData


def _write_ppi(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "protein_protein.csv").write_text(
        "proteinA_entrezid,proteinB_entrezid,databases\n"
        "7157,1956,BioGRID|IntAct\n"
        "7157,99999,BioGRID\n",
        encoding="utf-8",
    )


class TestPpiLoader:
    """Both endpoints harmonize to Ensembl; unmapped interactions are dropped."""

    def test_build_frames(self, reference_dir: Path) -> None:
        """Mapped edges become GEN-GEN INTERACTS_WITH; unmapped endpoints drop."""
        src = reference_dir / "ppi"
        _write_ppi(src)
        reference = ReferenceData(reference_dir)
        loader = PpiGraphLoader(
            src, harmonizer=Harmonizer(reference), enricher=NodeEnricher(reference)
        )
        nodes, edges = loader.build_frames()

        assert edges.height == 1
        edge = edges.row(0, named=True)
        assert {edge["from"], edge["to"]} == {"ENSG1", "ENSG2"}
        assert edge["type"] == "INTERACTS_WITH"
        assert edge["label"] == "GEN-GEN"
        assert edge["databases"] == ["BioGRID", "IntAct"]
        assert set(nodes["curie"].to_list()) == {"ENSG1", "ENSG2"}

    def test_schema_covers_emitted_types(self, reference_dir: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        src = reference_dir / "ppi"
        _write_ppi(src)
        reference = ReferenceData(reference_dir)
        loader = PpiGraphLoader(
            src, harmonizer=Harmonizer(reference), enricher=NodeEnricher(reference)
        )
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
