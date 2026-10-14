"""Tests for the GraphLoader base: seam mapping, streaming, and Parquet export.

A tiny in-test subclass supplies two polars frames; the assertions pin the base's
generic behavior — deterministic CURIE-derived ids, pre-linked relations,
property spreading with a dataset stamp, batch endpoint inclusion, the whole-graph
accessor, and round-trippable export.
"""

from __future__ import annotations

import polars as pl

from am_diag.common.data_models import KnowledgeGraph
from am_diag.common.data_models.schema import GraphSchema
from am_diag.loaders.graph.base import GraphLoader
from am_diag.loaders.graph.identity import curie_to_entity_id, relation_id


class _FakeGraphLoader(GraphLoader):
    """In-test loader returning fixed node/edge frames."""

    dataset_name = "fake"
    graph_schema = GraphSchema()

    def __init__(self, nodes: pl.DataFrame, edges: pl.DataFrame) -> None:
        self._nodes = nodes
        self._edges = edges

    def build_frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        return self._nodes, self._edges


def _nodes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "curie": ["CHEMBL25", "ENSG1", "ENSG2"],
            "name": ["aspirin", "BRCA1", "TP53"],
            "label": ["Drug", "Gene", "Gene"],
            "biotype": [None, "protein_coding", "protein_coding"],
        }
    )


def _edges() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "from": ["CHEMBL25", "CHEMBL25"],
            "to": ["ENSG1", "ENSG2"],
            "type": ["TARGET", "ENZYME"],
            "label": ["DRG-GEN", "DRG-GEN"],
            "sources": [["DRUG_BANK"], ["OPEN_TARGETS"]],
            "expression_rank": [1, 2],
        }
    )


async def _collect(loader: GraphLoader, **kwargs: object) -> list[KnowledgeGraph]:
    return [kg async for kg in loader.astream(**kwargs)]  # type: ignore[arg-type]


class TestGraphLoaderBase:
    """Seam-to-Entity/Relation mapping, streaming, and export."""

    async def test_astream_single_batch_has_all_entities_and_relations(self) -> None:
        """One batch (large batch_size) yields all endpoints and relations."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        batches = await _collect(loader, batch_size=10)
        assert len(batches) == 1
        kg = batches[0]
        assert len(kg.relations) == 2
        assert {e.curie for e in kg.entities} == {"CHEMBL25", "ENSG1", "ENSG2"}

    async def test_entity_id_is_curie_derived(self) -> None:
        """Entity ids are the deterministic UUID5 of their CURIE."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        kg = (await _collect(loader, batch_size=10))[0]
        by_curie = {e.curie: e for e in kg.entities}
        assert by_curie["ENSG1"].id == curie_to_entity_id("ENSG1")
        assert by_curie["ENSG1"].label == "Gene"
        assert by_curie["ENSG1"].name == "BRCA1"

    async def test_relation_is_pre_linked_and_deterministic(self) -> None:
        """Relations carry deterministic ids and CURIE-derived endpoints."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        kg = (await _collect(loader, batch_size=10))[0]
        rel = next(r for r in kg.relations if r.type == "TARGET")
        assert rel.id == relation_id("CHEMBL25", "TARGET", "ENSG1")
        assert rel.head_id == curie_to_entity_id("CHEMBL25")
        assert rel.tail_id == curie_to_entity_id("ENSG1")
        assert rel.head_name == "aspirin"
        assert rel.tail_name == "BRCA1"
        assert rel.structural_label == "DRG-GEN"
        assert rel.sources == ["DRUG_BANK"]

    async def test_non_reserved_columns_become_properties_with_stamp(self) -> None:
        """Extra columns land in schema_properties with the dataset stamp."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        kg = (await _collect(loader, batch_size=10))[0]
        gene = next(e for e in kg.entities if e.curie == "ENSG1")
        assert gene.schema_properties == {
            "biotype": "protein_coding",
            "dataset": "fake",
        }
        rel = next(r for r in kg.relations if r.type == "TARGET")
        assert rel.schema_properties == {"expression_rank": 1, "dataset": "fake"}

    async def test_none_property_is_dropped(self) -> None:
        """A null property column is not carried onto the entity."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        kg = (await _collect(loader, batch_size=10))[0]
        drug = next(e for e in kg.entities if e.curie == "CHEMBL25")
        assert "biotype" not in drug.schema_properties

    async def test_astream_batches_include_their_endpoints(self) -> None:
        """With batch_size=1 each batch carries exactly its edge's endpoints."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        batches = await _collect(loader, batch_size=1)
        assert len(batches) == 2
        for kg in batches:
            assert len(kg.relations) == 1
            assert len(kg.entities) == 2

    async def test_load_graph_returns_full_graph(self) -> None:
        """load_graph returns every node and edge in one KnowledgeGraph."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        kg = await loader.load_graph()
        assert len(kg.entities) == 3
        assert len(kg.relations) == 2

    def test_export_to_parquet_round_trips(self, tmp_path) -> None:
        """Exported frames are readable Parquet matching the source frames."""
        loader = _FakeGraphLoader(_nodes(), _edges())
        loader.export_to_parquet(tmp_path)
        nodes = pl.read_parquet(tmp_path / "nodes.parquet")
        edges = pl.read_parquet(tmp_path / "edges.parquet")
        assert nodes.height == 3
        assert edges.height == 2
        assert set(nodes["curie"].to_list()) == {"CHEMBL25", "ENSG1", "ENSG2"}
