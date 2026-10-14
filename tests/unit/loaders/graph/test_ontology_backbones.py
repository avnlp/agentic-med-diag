"""Tests for the single-type OBO backbone loaders (Uberon, Mondo, HPO).

A tiny OBO Graph JSON fixture per prefix drives the shared transform: terms become
identity nodes, ``is_a`` edges become ``PARENT`` relations, and non-matching
prefixes / predicates are filtered out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from am_diag.loaders.graph.identity import curie_to_entity_id
from am_diag.loaders.graph.ontology_backbones import (
    HpoGraphLoader,
    MondoGraphLoader,
    UberonGraphLoader,
)


_OBO = "http://purl.obolibrary.org/obo/"


def _write_backbone_obo(path: Path, prefix: str) -> None:
    graph = {
        "graphs": [
            {
                "nodes": [
                    {
                        "id": f"{_OBO}{prefix}0000001",
                        "lbl": "child term",
                        "meta": {
                            "definition": {"val": "a child"},
                            "synonyms": [{"val": "kid"}],
                            "xrefs": [{"val": "X:1"}],
                        },
                    },
                    {"id": f"{_OBO}{prefix}0000002", "lbl": "parent term", "meta": {}},
                    {"id": f"{_OBO}CL_0000999", "lbl": "off-prefix", "meta": {}},
                ],
                "edges": [
                    {
                        "sub": f"{_OBO}{prefix}0000001",
                        "pred": "is_a",
                        "obj": f"{_OBO}{prefix}0000002",
                    },
                    {
                        "sub": f"{_OBO}{prefix}0000001",
                        "pred": "part_of",
                        "obj": f"{_OBO}{prefix}0000002",
                    },
                ],
            }
        ]
    }
    path.write_text(json.dumps(graph), encoding="utf-8")


class TestOntologyBackbones:
    """build_frames mapping and schema consistency for the OBO backbones."""

    @pytest.mark.parametrize(
        ("loader_cls", "filename", "prefix", "label", "edge_label"),
        [
            (
                UberonGraphLoader,
                "uberon.json",
                "UBERON_",
                "AnatomicalStructure",
                "ANA-ANA",
            ),
            (MondoGraphLoader, "mondo.json", "MONDO_", "Disease", "DIS-DIS"),
            (HpoGraphLoader, "hp.json", "HP_", "Phenotype", "PHE-PHE"),
        ],
    )
    def test_build_frames(
        self,
        tmp_path: Path,
        loader_cls: type,
        filename: str,
        prefix: str,
        label: str,
        edge_label: str,
    ) -> None:
        """Terms map to nodes, is_a maps to PARENT, off-prefix rows are dropped."""
        _write_backbone_obo(tmp_path / filename, prefix)
        loader = loader_cls(tmp_path)
        nodes, edges = loader.build_frames()

        assert set(nodes["curie"].to_list()) == {
            f"{prefix}0000001",
            f"{prefix}0000002",
        }
        assert nodes["label"].unique().to_list() == [label]

        assert edges.height == 1
        edge = edges.row(0, named=True)
        assert edge["from"] == f"{prefix}0000002"
        assert edge["to"] == f"{prefix}0000001"
        assert edge["type"] == "PARENT"
        assert edge["label"] == edge_label
        assert edge["sources"] == [loader_cls.source.value]

        # Emitted types are covered by the authored schema.
        schema = loader.graph_schema
        assert set(nodes["label"].to_list()) <= schema.entity_type_names()
        assert {edge["type"]} <= schema.relation_type_names()

    async def test_load_graph_assigns_deterministic_ids(self, tmp_path: Path) -> None:
        """load_graph yields a KnowledgeGraph with CURIE-derived entity ids."""
        _write_backbone_obo(tmp_path / "uberon.json", "UBERON_")
        kg = await UberonGraphLoader(tmp_path).load_graph()
        ids = {e.id for e in kg.entities}
        assert curie_to_entity_id("UBERON_0000001") in ids
        assert len(kg.relations) == 1
