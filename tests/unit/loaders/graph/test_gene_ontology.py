"""Tests for the Gene Ontology loader: namespace-split labels and is_a edges."""

from __future__ import annotations

import json
from pathlib import Path

from am_diag.loaders.graph.gene_ontology import GeneOntologyGraphLoader


_OBO = "http://purl.obolibrary.org/obo/"
_HAS_NS = "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"


def _go_node(go_id: str, name: str, namespace: str) -> dict:
    return {
        "id": f"{_OBO}{go_id}",
        "lbl": name,
        "meta": {
            "basicPropertyValues": [{"pred": _HAS_NS, "val": namespace}],
            "definition": {"val": f"{name} def"},
            "synonyms": [{"val": name}],
        },
    }


def _write_go(path: Path) -> None:
    graph = {
        "graphs": [
            {
                "nodes": [
                    _go_node("GO_0008150", "biological process", "biological_process"),
                    _go_node("GO_0009987", "cellular process", "biological_process"),
                    _go_node("GO_0005575", "cellular component", "cellular_component"),
                ],
                "edges": [
                    {
                        "sub": f"{_OBO}GO_0009987",
                        "pred": "is_a",
                        "obj": f"{_OBO}GO_0008150",
                    }
                ],
            }
        ]
    }
    path.write_text(json.dumps(graph), encoding="utf-8")


class TestGeneOntologyLoader:
    """Namespace drives the entity label and the within-namespace edge label."""

    def test_build_frames_splits_namespaces(self, tmp_path: Path) -> None:
        """Each term takes its namespace's label; is_a stays within a namespace."""
        _write_go(tmp_path / "go_plus.json")
        nodes, edges = GeneOntologyGraphLoader(tmp_path).build_frames()

        labels = dict(
            zip(nodes["curie"].to_list(), nodes["label"].to_list(), strict=True)
        )
        assert labels["GO_0008150"] == "GoBiologicalProcess"
        assert labels["GO_0009987"] == "GoBiologicalProcess"
        assert labels["GO_0005575"] == "GoCellularComponent"

        assert edges.height == 1
        edge = edges.row(0, named=True)
        assert edge["from"] == "GO_0008150"
        assert edge["to"] == "GO_0009987"
        assert edge["type"] == "PARENT"
        assert edge["label"] == "BPO-BPO"

    def test_schema_covers_emitted_types(self, tmp_path: Path) -> None:
        """All emitted labels and relation types are in the authored schema."""
        _write_go(tmp_path / "go_plus.json")
        loader = GeneOntologyGraphLoader(tmp_path)
        nodes, edges = loader.build_frames()
        assert set(nodes["label"].to_list()) <= loader.graph_schema.entity_type_names()
        assert set(edges["type"].to_list()) <= loader.graph_schema.relation_type_names()
