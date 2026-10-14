"""Tests for the DisGeNET disease/phenotype-gene loader."""

from __future__ import annotations

from pathlib import Path

from am_diag.loaders.graph.disgenet import DisGeNETGraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData


def _write_disgenet(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        "geneSymbol\tdiseaseId\tdiseaseName\tdiseaseType\tscore\tEI",
        "TP53\tC0011860\ttype 2 diabetes\tdisease\t0.9\t0.8",
        "EGFR\tC0011860\ttype 2 diabetes\tphenotype\t0.5\t0.4",
    ]
    (directory / "curated_gene_disease_associations.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


class TestDisGeNETLoader:
    """Symbol->Ensembl and UMLS->canonical mapping with DIS/PHE-GEN split."""

    def test_build_frames(self, reference_dir: Path) -> None:
        """Rows split into DIS-GEN/PHE-GEN with harmonized endpoints and scores."""
        src = reference_dir / "disgenet"
        _write_disgenet(src)
        reference = ReferenceData(reference_dir)
        loader = DisGeNETGraphLoader(
            src,
            reference=reference,
            harmonizer=Harmonizer(reference),
            enricher=NodeEnricher(reference),
        )
        nodes, edges = loader.build_frames()

        assert edges.height == 2
        by_to = {r["to"]: r for r in edges.iter_rows(named=True)}
        assert by_to["ENSG1"]["from"] == "MONDO_0005148"
        assert by_to["ENSG1"]["label"] == "DIS-GEN"
        assert by_to["ENSG1"]["type"] == "ASSOCIATED_WITH"
        assert by_to["ENSG1"]["disgenet_score"] == 0.9
        assert by_to["ENSG2"]["label"] == "PHE-GEN"

        disease = nodes.filter(nodes["curie"] == "MONDO_0005148").row(0, named=True)
        assert disease["name"] == "type 2 diabetes"

    def test_schema_covers_emitted_types(self, reference_dir: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        src = reference_dir / "disgenet"
        _write_disgenet(src)
        reference = ReferenceData(reference_dir)
        loader = DisGeNETGraphLoader(
            src,
            reference=reference,
            harmonizer=Harmonizer(reference),
            enricher=NodeEnricher(reference),
        )
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
