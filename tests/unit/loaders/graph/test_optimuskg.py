"""Tests for the released-OptimusKG loader's gold Parquet -> seam mapping.

Fixtures write a tiny gold directory in the shape of the published release, in both
supported layouts: per-type files whose ``properties`` column is a native struct, and
consolidated files whose ``properties`` column is a JSON string. In both, edges carry
``relation`` and ``undirected`` as top-level columns, and ``sources`` is a
``{direct, indirect}`` struct. Assertions pin the OptimusKG-specific mapping:
``id`` -> ``curie``, abbreviation -> PascalCase label, ``relation`` -> ``type``,
``sources`` -> a list of dataset names, ``relation_assertions`` -> a flat
``asserted_relations`` list, and ``undirected`` kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from am_diag.loaders.graph.identity import curie_to_entity_id
from am_diag.loaders.graph.optimuskg import OptimusKGGraphLoader


def _sources(direct: list[str], indirect: list[str] | None = None) -> dict[str, Any]:
    return {"direct": direct, "indirect": indirect or []}


_NODES: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "GEN": [
        (
            "ENSG1",
            {
                "name": "BRCA1",
                "biotype": "protein_coding",
                "synonyms": [{"label": "BRCA-1", "source": "uniprot"}],
                "xrefs": [{"id": "1100", "source": "HGNC"}],
                "sources": _sources(["OPEN_TARGETS"]),
            },
        )
    ],
    "DRG": [
        (
            "CHEMBL25",
            {
                "name": "aspirin",
                "description": "a drug",
                "inchi_key": "X",
                "synonyms": ["ASA"],
                "xrefs": ["DrugBank:DB00945"],
                "mol_file_base64": "bW9s",
                "mol_image_base64": "aW1n",
                "sources": _sources(["DRUG_BANK"], ["ONSIDES"]),
            },
        )
    ],
    "DIS": [("MONDO:1", {"name": "disease one", "sources": _sources(["MONDO"])})],
}

# (from, to, relation, undirected, properties) grouped by structural label.
_EDGES: dict[str, list[tuple[str, str, str, bool, dict[str, Any]]]] = {
    "DRG-GEN": [
        (
            "CHEMBL25",
            "ENSG1",
            "INHIBITOR",
            False,
            {
                "sources": _sources(["DRUG_BANK"]),
                "mechanisms_of_action": "inhibitor",
            },
        )
    ],
    "DRG-DIS": [
        (
            "CHEMBL25",
            "MONDO:1",
            "INDICATION",
            True,
            {
                "sources": _sources(["DRUG_BANK", "OPEN_TARGETS"]),
                "relation_assertions": [
                    {"source": "DRUG_BANK", "relation": "INDICATION"},
                    {"source": "OPEN_TARGETS", "relation": "CONTRAINDICATION"},
                ],
                "relation_conflict": True,
            },
        )
    ],
    "DIS-GEN": [
        (
            "MONDO:1",
            "ENSG1",
            "ASSOCIATED_WITH",
            True,
            {"sources": _sources(["OPEN_TARGETS"]), "score": 0.8},
        )
    ],
}

_NODE_FILES = {"GEN": "gene", "DRG": "drug", "DIS": "disease"}
_EDGE_FILES = {
    "DRG-GEN": "drug_gene",
    "DRG-DIS": "drug_disease",
    "DIS-GEN": "disease_gene",
}


def _edge_frame(label: str, properties: list[Any]) -> pl.DataFrame:
    rows = _EDGES[label]
    return pl.DataFrame(
        {
            "from": [r[0] for r in rows],
            "to": [r[1] for r in rows],
            "label": [label] * len(rows),
            "relation": [r[2] for r in rows],
            "undirected": [r[3] for r in rows],
            "properties": properties,
        }
    )


def _write_per_type_gold(root: Path) -> None:
    (root / "nodes").mkdir(parents=True)
    (root / "edges").mkdir(parents=True)
    for label, rows in _NODES.items():
        pl.DataFrame(
            {
                "id": [r[0] for r in rows],
                "label": [label] * len(rows),
                "properties": [r[1] for r in rows],
            }
        ).write_parquet(root / "nodes" / f"{_NODE_FILES[label]}.parquet")
    for label, rows in _EDGES.items():
        _edge_frame(label, [r[4] for r in rows]).write_parquet(
            root / "edges" / f"{_EDGE_FILES[label]}.parquet"
        )


def _write_consolidated_gold(root: Path) -> None:
    node_rows = [
        (node_id, label, json.dumps(props))
        for label, rows in _NODES.items()
        for node_id, props in rows
    ]
    pl.DataFrame(
        {
            "id": [r[0] for r in node_rows],
            "label": [r[1] for r in node_rows],
            "properties": [r[2] for r in node_rows],
        }
    ).write_parquet(root / "nodes.parquet")
    pl.concat(
        [
            _edge_frame(label, [json.dumps(r[4]) for r in rows])
            for label, rows in _EDGES.items()
        ]
    ).write_parquet(root / "edges.parquet")


@pytest.fixture(params=["per_type", "consolidated"])
def gold_dir(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """Write the gold fixture in each supported layout and return its directory."""
    if request.param == "per_type":
        _write_per_type_gold(tmp_path)
    else:
        _write_consolidated_gold(tmp_path)
    return tmp_path


class TestOptimusKGLoader:
    """Mapping of the gold release shape, identical across both layouts."""

    async def test_nodes_mapped_to_pascalcase_with_curie_ids(self, gold_dir) -> None:
        """Node abbreviations become PascalCase labels and ids are CURIE-derived."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        by_curie = {e.curie: e for e in kg.entities}
        assert by_curie["ENSG1"].label == "Gene"
        assert by_curie["ENSG1"].id == curie_to_entity_id("ENSG1")
        assert by_curie["ENSG1"].name == "BRCA1"
        assert by_curie["CHEMBL25"].label == "Drug"

    async def test_node_description_maps_to_core_field(self, gold_dir) -> None:
        """A drug ``description`` property populates Entity.description, no clash."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        drug = next(e for e in kg.entities if e.curie == "CHEMBL25")
        assert drug.description == "a drug"
        assert "description" not in drug.schema_properties

    async def test_node_sources_are_a_list_with_indirect_sources(
        self, gold_dir
    ) -> None:
        """Node ``sources`` is a queryable list and indirect sources are kept apart."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        drug = next(e for e in kg.entities if e.curie == "CHEMBL25")
        assert drug.schema_properties["sources"] == ["DRUG_BANK"]
        assert drug.schema_properties["indirect_sources"] == ["ONSIDES"]

    async def test_gene_synonyms_and_xrefs_match_the_other_node_types(
        self, gold_dir
    ) -> None:
        """Gene entries given as objects load as the string lists other types use."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        by_curie = {e.curie: e for e in kg.entities}
        gene, drug = by_curie["ENSG1"], by_curie["CHEMBL25"]
        assert gene.schema_properties["synonyms"] == ["BRCA-1"]
        assert gene.schema_properties["xrefs"] == ["HGNC:1100"]
        assert drug.schema_properties["synonyms"] == ["ASA"]
        assert drug.schema_properties["xrefs"] == ["DrugBank:DB00945"]

    async def test_drug_structure_blobs_are_not_stored(self, gold_dir) -> None:
        """Base64 molecule files and images are dropped; other properties remain."""
        nodes, _ = OptimusKGGraphLoader(gold_dir).build_frames()
        assert "mol_file_base64" not in nodes.columns
        assert "mol_image_base64" not in nodes.columns
        assert "inchi_key" in nodes.columns

    async def test_edge_relation_and_structural_label(self, gold_dir) -> None:
        """``relation`` becomes type and the structural label is retained."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        rel = next(r for r in kg.relations if r.type == "INHIBITOR")
        assert rel.structural_label == "DRG-GEN"
        assert rel.head_id == curie_to_entity_id("CHEMBL25")
        assert rel.schema_properties["undirected"] is False
        assert rel.schema_properties["mechanisms_of_action"] == "inhibitor"

    async def test_edge_sources_are_dataset_names_not_a_stringified_struct(
        self, gold_dir
    ) -> None:
        """``Relation.sources`` holds the direct source names as a list of strings."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        rel = next(r for r in kg.relations if r.type == "INHIBITOR")
        assert rel.sources == ["DRUG_BANK"]
        indication = next(r for r in kg.relations if r.type == "INDICATION")
        assert indication.sources == ["DRUG_BANK", "OPEN_TARGETS"]

    async def test_conflicting_assertions_are_flattened_for_filtering(
        self, gold_dir
    ) -> None:
        """A drug-disease pair asserted two ways exposes both relations as a list."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        rel = next(r for r in kg.relations if r.type == "INDICATION")
        assert rel.schema_properties["asserted_relations"] == [
            "CONTRAINDICATION",
            "INDICATION",
        ]
        assert rel.schema_properties["relation_conflict"] is True
        assert rel.schema_properties["undirected"] is True

    async def test_edges_without_assertions_have_no_asserted_relations(
        self, gold_dir
    ) -> None:
        """Edges without ``relation_assertions`` get no ``asserted_relations``."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        rel = next(r for r in kg.relations if r.type == "INHIBITOR")
        assert "asserted_relations" not in rel.schema_properties

    async def test_edge_score_maps_to_core_field(self, gold_dir) -> None:
        """A DisGeNET-style ``score`` property populates Relation.score."""
        kg = await OptimusKGGraphLoader(gold_dir).load_graph()
        rel = next(r for r in kg.relations if r.type == "ASSOCIATED_WITH")
        assert rel.score == 0.8
        assert "score" not in rel.schema_properties


class TestOptimusKGLoaderLargestConnectedComponent:
    """The largest-connected-component release variant, selected explicitly."""

    @staticmethod
    def _write_lcc_files(root: Path) -> None:
        pl.DataFrame(
            {
                "id": ["ENSG1", "CHEMBL25"],
                "label": ["GEN", "DRG"],
                "properties": [json.dumps({"name": "BRCA1"}), json.dumps({})],
            }
        ).write_parquet(root / "largest_connected_component_nodes.parquet")
        pl.DataFrame(
            {
                "from": ["CHEMBL25"],
                "to": ["ENSG1"],
                "label": ["DRG-GEN"],
                "relation": ["INHIBITOR"],
                "undirected": [False],
                "properties": [json.dumps({"sources": _sources(["DRUG_BANK"])})],
            }
        ).write_parquet(root / "largest_connected_component_edges.parquet")

    async def test_reads_only_the_component_files_when_requested(
        self, tmp_path
    ) -> None:
        """With the option set, the full graph files in the directory are ignored."""
        _write_consolidated_gold(tmp_path)
        self._write_lcc_files(tmp_path)
        loader = OptimusKGGraphLoader(tmp_path, largest_connected_component=True)
        kg = await loader.load_graph()
        assert {e.curie for e in kg.entities} == {"ENSG1", "CHEMBL25"}
        assert [r.type for r in kg.relations] == ["INHIBITOR"]

    async def test_defaults_to_the_full_graph(self, tmp_path) -> None:
        """Without the option, the component files do not shadow the full graph."""
        _write_consolidated_gold(tmp_path)
        self._write_lcc_files(tmp_path)
        kg = await OptimusKGGraphLoader(tmp_path).load_graph()
        assert {e.curie for e in kg.entities} == {"ENSG1", "CHEMBL25", "MONDO:1"}

    def test_missing_component_files_raise(self, tmp_path) -> None:
        """Requesting the component variant fails clearly when its files are absent."""
        _write_consolidated_gold(tmp_path)
        loader = OptimusKGGraphLoader(tmp_path, largest_connected_component=True)
        with pytest.raises(FileNotFoundError, match="largest_connected_component"):
            loader.build_frames()
