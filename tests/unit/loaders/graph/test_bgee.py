"""Tests for the Bgee anatomy-gene expression loader."""

from __future__ import annotations

import gzip
from pathlib import Path

from am_diag.loaders.graph.bgee import BgeeGraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.reference_data import ReferenceData


_HEADER = (
    "Gene ID\tGene name\tAnatomical entity ID\tAnatomical entity name\t"
    "Expression\tCall quality\tExpression rank"
)


def _write_bgee(path: Path) -> None:
    rows = [
        _HEADER,
        "ENSG1\tTP53\tUBERON:0002107\tliver\tpresent\tgold quality\t1000",
        "ENSG2\tEGFR\tUBERON:0002107\tliver\tabsent\tsilver quality\t2000",
        "ENSG1\tTP53\tUBERON:0002107 ∩ UBERON:0000955\tcomposite\tpresent\tgold\t5",
        "ENSG1\tTP53\tEFO:0001\tnot anatomy\tpresent\tgold\t5",
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")


class TestBgeeLoader:
    """Expression edges, the UBERON/composite filters, and gene enrichment."""

    def test_build_frames(self, reference_dir: Path) -> None:
        """Present/absent edges map; composite and non-Uberon rows are dropped."""
        _write_bgee(reference_dir / "Homo_sapiens_expr_advanced.tsv.gz")
        loader = BgeeGraphLoader(
            reference_dir, enricher=NodeEnricher(ReferenceData(reference_dir))
        )
        nodes, edges = loader.build_frames()

        assert edges.height == 2
        by_to = {r["to"]: r for r in edges.iter_rows(named=True)}
        assert by_to["ENSG1"]["type"] == "EXPRESSION_PRESENT"
        assert by_to["ENSG2"]["type"] == "EXPRESSION_ABSENT"
        assert by_to["ENSG1"]["from"] == "UBERON_0002107"
        assert by_to["ENSG1"]["label"] == "ANA-GEN"
        assert by_to["ENSG1"]["expression_rank"] == 1000

        gene = nodes.filter(nodes["curie"] == "ENSG1").row(0, named=True)
        assert gene["label"] == "Gene"
        assert gene["symbol"] == "TP53"
        anatomy = nodes.filter(nodes["curie"] == "UBERON_0002107").row(0, named=True)
        assert anatomy["label"] == "AnatomicalStructure"
        assert anatomy["name"] == "liver"

    def test_schema_covers_emitted_types(self, reference_dir: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        _write_bgee(reference_dir / "Homo_sapiens_expr_advanced.tsv.gz")
        loader = BgeeGraphLoader(
            reference_dir, enricher=NodeEnricher(ReferenceData(reference_dir))
        )
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
