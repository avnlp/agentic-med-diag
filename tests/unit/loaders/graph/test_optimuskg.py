"""Tests for the released-OptimusKG loader's gold Parquet -> seam mapping.

Fixtures write tiny gold files in both supported layouts (per-type struct files
and consolidated JSON-property files). Assertions pin the OptimusKG-specific
mapping: ``id``->``curie``, ``input_label``->PascalCase, ``relation``->``type``,
``undirected`` kept, and core-field properties (``description``, ``score``,
``sources``) mapped through the seam rather than colliding.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from am_diag.loaders.graph.identity import curie_to_entity_id
from am_diag.loaders.graph.optimuskg import OptimusKGGraphLoader


def _write_per_type_gold(root: Path) -> None:
    (root / "nodes").mkdir(parents=True)
    (root / "edges").mkdir(parents=True)
    pl.DataFrame(
        {
            "id": ["ENSG1"],
            "label": ["GEN"],
            "properties": [
                {"name": "BRCA1", "biotype": "protein_coding", "sources": "opentargets"}
            ],
        }
    ).write_parquet(root / "nodes" / "gene.parquet")
    pl.DataFrame(
        {
            "id": ["CHEMBL25"],
            "label": ["DRG"],
            "properties": [
                {"name": "aspirin", "description": "a drug", "inchi_key": "X"}
            ],
        }
    ).write_parquet(root / "nodes" / "drug.parquet")
    pl.DataFrame(
        {
            "id": ["MONDO:1"],
            "label": ["DIS"],
            "properties": [{"name": "disease one"}],
        }
    ).write_parquet(root / "nodes" / "disease.parquet")
    pl.DataFrame(
        {
            "from": ["CHEMBL25"],
            "to": ["ENSG1"],
            "label": ["DRG-GEN"],
            "properties": [
                {
                    "relation": "TARGET",
                    "undirected": False,
                    "sources": "drugbank",
                    "mechanisms_of_action": "inhibitor",
                }
            ],
        }
    ).write_parquet(root / "edges" / "drug_gene.parquet")
    pl.DataFrame(
        {
            "from": ["MONDO:1"],
            "to": ["ENSG1"],
            "label": ["DIS-GEN"],
            "properties": [
                {"relation": "ASSOCIATED_WITH", "undirected": False, "score": 0.8}
            ],
        }
    ).write_parquet(root / "edges" / "disease_gene.parquet")


class TestOptimusKGLoaderPerType:
    """Per-type struct layout (nodes/*.parquet, edges/*.parquet)."""

    async def test_nodes_mapped_to_pascalcase_with_curie_ids(self, tmp_path) -> None:
        """Node input_labels become PascalCase and ids are CURIE-derived."""
        _write_per_type_gold(tmp_path)
        kg = await OptimusKGGraphLoader(tmp_path).load_graph()
        by_curie = {e.curie: e for e in kg.entities}
        assert by_curie["ENSG1"].label == "Gene"
        assert by_curie["ENSG1"].id == curie_to_entity_id("ENSG1")
        assert by_curie["ENSG1"].name == "BRCA1"
        assert by_curie["CHEMBL25"].label == "Drug"

    async def test_node_description_maps_to_core_field(self, tmp_path) -> None:
        """A drug ``description`` property populates Entity.description, no clash."""
        _write_per_type_gold(tmp_path)
        kg = await OptimusKGGraphLoader(tmp_path).load_graph()
        drug = next(e for e in kg.entities if e.curie == "CHEMBL25")
        assert drug.description == "a drug"
        assert "description" not in drug.schema_properties

    async def test_edge_relation_and_structural_label(self, tmp_path) -> None:
        """``relation`` becomes type; structural label retained; undirected kept."""
        _write_per_type_gold(tmp_path)
        kg = await OptimusKGGraphLoader(tmp_path).load_graph()
        rel = next(r for r in kg.relations if r.type == "TARGET")
        assert rel.structural_label == "DRG-GEN"
        assert rel.head_id == curie_to_entity_id("CHEMBL25")
        assert rel.sources == ["drugbank"]
        assert rel.schema_properties["undirected"] is False
        assert rel.schema_properties.get("mechanisms_of_action") == "inhibitor"

    async def test_edge_score_maps_to_core_field(self, tmp_path) -> None:
        """A DisGeNET-style ``score`` property populates Relation.score."""
        _write_per_type_gold(tmp_path)
        kg = await OptimusKGGraphLoader(tmp_path).load_graph()
        rel = next(r for r in kg.relations if r.type == "ASSOCIATED_WITH")
        assert rel.score == 0.8
        assert "score" not in rel.schema_properties


class TestOptimusKGLoaderConsolidated:
    """Consolidated layout (nodes.parquet, edges.parquet, JSON properties)."""

    async def test_json_properties_are_decoded(self, tmp_path) -> None:
        """JSON-string properties in consolidated files flatten the same way."""
        pl.DataFrame(
            {
                "id": ["ENSG1", "CHEMBL25"],
                "label": ["GEN", "DRG"],
                "properties": [
                    json.dumps({"name": "BRCA1", "biotype": "pc"}),
                    json.dumps({"name": "aspirin"}),
                ],
            }
        ).write_parquet(tmp_path / "nodes.parquet")
        pl.DataFrame(
            {
                "from": ["CHEMBL25"],
                "to": ["ENSG1"],
                "label": ["DRG-GEN"],
                "properties": [json.dumps({"relation": "TARGET", "undirected": False})],
            }
        ).write_parquet(tmp_path / "edges.parquet")
        kg = await OptimusKGGraphLoader(tmp_path).load_graph()
        gene = next(e for e in kg.entities if e.curie == "ENSG1")
        assert gene.label == "Gene"
        assert gene.schema_properties.get("biotype") == "pc"
        rel = kg.relations[0]
        assert rel.type == "TARGET"
        assert rel.structural_label == "DRG-GEN"
