"""Tests for the DrugCentral drug-disease loader (serverless pg_dump parsing)."""

from __future__ import annotations

import gzip
from pathlib import Path

from am_diag.loaders.graph.drugcentral import DrugCentralGraphLoader
from am_diag.loaders.graph.enricher import NodeEnricher
from am_diag.loaders.graph.harmonizer import Harmonizer
from am_diag.loaders.graph.reference_data import ReferenceData


def _write_dump(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dump = (
        "COPY public.identifier (struct_id, identifier, id_type) FROM stdin;\n"
        "1\tCHEMBL25\tChEMBL_ID\n"
        "1\tDB00945\tDRUGBANK_ID\n"
        "\\.\n"
        "COPY public.omop_relationship "
        "(struct_id, relationship_name, umls_cui, concept_name) FROM stdin;\n"
        "1\tindication\tC0011860\ttype 2 diabetes\n"
        "1\tcontraindication\tC0011860\ttype 2 diabetes\n"
        "1\toff-label use\tC0011860\ttype 2 diabetes\n"
        "1\tindication\tC9999\tunmapped disease\n"
        "\\.\n"
    )
    with gzip.open(directory / "drugcentral_dump.sql.gz", "wt", encoding="utf-8") as fh:
        fh.write(dump)


class TestDrugCentralLoader:
    """ChEMBL + UMLS->canonical mapping; a pair becomes one edge; unmapped rows drop."""

    def test_build_frames(self, reference_dir: Path) -> None:
        """A pair with three relations becomes one DRG-DIS edge with harmonized ids."""
        src = reference_dir / "drugcentral"
        _write_dump(src)
        reference = ReferenceData(reference_dir)
        loader = DrugCentralGraphLoader(
            src, harmonizer=Harmonizer(reference), enricher=NodeEnricher(reference)
        )
        nodes, edges = loader.build_frames()

        assert edges.height == 1
        edge = edges.row(0, named=True)
        assert edge["from"] == "CHEMBL25"
        assert edge["to"] == "MONDO_0005148"
        assert edge["label"] == "DRG-DIS"
        assert edge["type"] == "INDICATION"
        assert edge["asserted_relations"] == [
            "CONTRAINDICATION",
            "INDICATION",
            "OFF_LABEL_USE",
        ]
        assert edge["relation_conflict"] is True
        assert edge["sources"] == ["DRUG_CENTRAL"]

        drug = nodes.filter(nodes["curie"] == "CHEMBL25").row(0, named=True)
        assert drug["name"] == "aspirin"
        disease = nodes.filter(nodes["curie"] == "MONDO_0005148").row(0, named=True)
        assert disease["label"] == "Disease"
        assert disease["name"] == "type 2 diabetes"

    def test_off_label_use_alone_is_kept(self, reference_dir: Path) -> None:
        """An off-label-only pair keeps its relation and raises no conflict."""
        src = reference_dir / "drugcentral"
        _write_dump(src)
        dump = src / "drugcentral_dump.sql.gz"
        with gzip.open(dump, "rt", encoding="utf-8") as fh:
            text = fh.read()
        text = text.replace("1\tindication\tC0011860", "1\tx\tC0011860").replace(
            "1\tcontraindication\tC0011860", "1\tx\tC0011860"
        )
        with gzip.open(dump, "wt", encoding="utf-8") as fh:
            fh.write(text)
        reference = ReferenceData(reference_dir)
        loader = DrugCentralGraphLoader(
            src, harmonizer=Harmonizer(reference), enricher=NodeEnricher(reference)
        )
        _, edges = loader.build_frames()

        edge = edges.row(0, named=True)
        assert edge["type"] == "OFF_LABEL_USE"
        assert edge["relation_conflict"] is False

    def test_schema_covers_emitted_types(self, reference_dir: Path) -> None:
        """Emitted labels and relation types fall within the authored schema."""
        src = reference_dir / "drugcentral"
        _write_dump(src)
        reference = ReferenceData(reference_dir)
        loader = DrugCentralGraphLoader(
            src, harmonizer=Harmonizer(reference), enricher=NodeEnricher(reference)
        )
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
