"""Tests for the raw-format readers (OBO JSON, DrugBank XML, zip CSV, pg_dump).

Fixtures are written to ``tmp_path`` (text inline, binary generated in-test) to keep
the multi-gigabyte real inputs out of the suite while still exercising the parsing
paths that the per-source loaders depend on.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

from am_diag.loaders.graph.formats import (
    obo_isa_edges,
    obo_terms,
    read_drugbank_interactions_xml,
    read_obograph_json,
    read_pg_dump_table,
    read_zip_csv,
)


_OBO = "http://purl.obolibrary.org/obo/"


def _write_obograph(path: Path) -> None:
    graph = {
        "graphs": [
            {
                "nodes": [
                    {
                        "id": f"{_OBO}UBERON_0002107",
                        "lbl": "liver",
                        "meta": {
                            "definition": {"val": "an organ"},
                            "synonyms": [{"val": "hepatic tissue"}],
                            "xrefs": [{"val": "FMA:7197"}],
                        },
                    }
                ],
                "edges": [
                    {
                        "sub": f"{_OBO}UBERON_0002107",
                        "pred": "is_a",
                        "obj": f"{_OBO}UBERON_0000062",
                    },
                    {
                        "sub": f"{_OBO}UBERON_0002107",
                        "pred": "part_of",
                        "obj": f"{_OBO}UBERON_0000949",
                    },
                ],
            }
        ]
    }
    path.write_text(json.dumps(graph), encoding="utf-8")


class TestOboGraph:
    """OBO Graph JSON term and is_a edge extraction."""

    def test_obo_terms_flattens_nodes(self, tmp_path: Path) -> None:
        """Term rows flatten name, definition, synonyms, and xrefs per node."""
        path = tmp_path / "uberon.json"
        _write_obograph(path)
        terms = obo_terms(read_obograph_json(path))
        row = terms.row(0, named=True)
        assert row["id"] == "UBERON_0002107"
        assert row["name"] == "liver"
        assert row["definition"] == "an organ"
        assert row["synonyms"] == ["hepatic tissue"]
        assert row["xrefs"] == ["FMA:7197"]

    def test_obo_isa_edges_keeps_only_is_a(self, tmp_path: Path) -> None:
        """Only ``is_a`` edges are emitted as parent/child pairs."""
        path = tmp_path / "uberon.json"
        _write_obograph(path)
        edges = obo_isa_edges(read_obograph_json(path))
        assert edges.height == 1
        row = edges.row(0, named=True)
        assert row["parent"] == "UBERON_0000062"
        assert row["child"] == "UBERON_0002107"


class TestDrugBankXml:
    """DrugBank drug-drug interaction XML streaming."""

    def test_extracts_sorted_interaction_pairs(self, tmp_path: Path) -> None:
        """Interaction pairs are emitted with sorted, prefixed DrugBank ids."""
        xml = (
            '<?xml version="1.0"?>'
            '<drugbank xmlns="http://www.drugbank.ca">'
            "<drug><drugbank-id>DB001</drugbank-id>"
            "<drug-interactions><drug-interaction>"
            "<drugbank-id>DB002</drugbank-id><description>desc12</description>"
            "</drug-interaction></drug-interactions></drug>"
            "<drug><drugbank-id>DB002</drugbank-id></drug>"
            "</drugbank>"
        )
        path = tmp_path / "full database.xml"
        path.write_text(xml, encoding="utf-8")
        out = read_drugbank_interactions_xml(path)
        assert out.height == 1
        row = out.row(0, named=True)
        assert row["tail_drug_id"] == "DRUGBANK:DB001"
        assert row["head_drug_id"] == "DRUGBANK:DB002"
        assert row["description"] == "desc12"


class TestReadZipCsv:
    """CSV extraction from a zip member."""

    def test_reads_member(self, tmp_path: Path) -> None:
        """A named CSV member is read out of the zip archive as a frame."""
        archive = tmp_path / "onsides.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("adverse.csv", "drug,effect\nDB1,nausea\n")
        out = read_zip_csv(archive, "adverse.csv")
        assert out.to_dicts() == [{"drug": "DB1", "effect": "nausea"}]


class TestReadPgDumpTable:
    """Serverless extraction of a COPY block from a gzipped pg_dump."""

    def test_parses_copy_block_with_nulls(self, tmp_path: Path) -> None:
        """The target COPY block is parsed, mapping the null sentinel to null."""
        dump = (
            "SET statement_timeout = 0;\n"
            "COPY public.structures (id, name) FROM stdin;\n"
            "1\taspirin\n"
            "2\t\\N\n"
            "\\.\n"
            "COPY public.other (x) FROM stdin;\n"
            "9\n"
            "\\.\n"
        )
        path = tmp_path / "drugcentral_dump.sql.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(dump)
        out = read_pg_dump_table(path, "public.structures", ["id", "name"])
        assert out.to_dicts() == [
            {"id": "1", "name": "aspirin"},
            {"id": "2", "name": None},
        ]
