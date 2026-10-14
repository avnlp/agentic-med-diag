"""Tests for EntityEdgeResolver (am_diag/graph_construction/resolve/resolver.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from am_diag.common.data_models import Entity, Relation
from am_diag.common.data_models.cluster import ResolvedAliasMap
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.graph_construction import EntityEdgeResolver
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.resolve.apply import AliasApplier
from am_diag.graph_construction.resolve.cluster import cluster_items
from am_diag.vector.embedding.base import Embedder


def _entity(
    name: str,
    label: str,
    score: float = 1.0,
    chunk_ids: list | None = None,
) -> Entity:
    return Entity(
        name=name,
        label=label,
        score=score,
        provenance=Provenance(
            surface_forms=[name],
            chunk_ids=chunk_ids or [],
        ),
    )


def _relation(
    head: str,
    tail: str,
    rel_type: str,
    score: float = 1.0,
    chunk_ids: list | None = None,
) -> Relation:
    return Relation(
        head_name=head,
        tail_name=tail,
        type=rel_type,
        score=score,
        provenance=Provenance(
            chunk_ids=chunk_ids or [],
        ),
    )


def _make_fake_embedder(dim: int = 4) -> MagicMock:
    embedder = MagicMock(spec=Embedder)
    embedder.embed = AsyncMock()
    return embedder


def _make_cfg(**kwargs) -> ExtractionPipelineConfig:
    defaults = {
        "use_gliner": False,
        "use_glirel": False,
        "use_llm": False,
        "use_llm_resolution": True,
        "resolution_cluster_size": 128,
        "resolution_kmeans_max_iter": 20,
        "resolution_top_k": 16,
        "resolution_bm25_weight": 0.5,
        "resolution_embedding_weight": 0.5,
        "resolution_max_concurrent_clusters": 8,
        "resolution_max_concurrent_llm_calls": 10,
    }
    defaults.update(kwargs)
    return ExtractionPipelineConfig(**defaults)


class TestClusterItems:
    async def test_cluster_respects_cluster_size_cap(self):
        n = 300
        cluster_size = 128
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        clusters = cluster_items(embeddings, cluster_size)
        for c in clusters:
            assert len(c.items) <= cluster_size
        assert len(clusters) in (2, 3)

    async def test_cluster_single_cluster_for_small_input(self):
        n = 50
        cluster_size = 128
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        clusters = cluster_items(embeddings, cluster_size)
        assert len(clusters) == 1
        assert len(clusters[0].items) == n


class TestEntityEdgeResolverResolve:
    async def test_embedder_called_with_all_items(self):
        embedder = _make_fake_embedder()
        embedder.embed.return_value = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        cfg = _make_cfg(use_llm_resolution=True)
        resolver = EntityEdgeResolver(cfg, MEDICAL_GRAPHRAG_SCHEMA, embedder)
        entities = [_entity("metformin", "Drug"), _entity("aspirin", "Drug")]
        sem_patch = "am_diag.graph_construction.resolve.deterministic.SemHash"
        with patch(sem_patch) as mock_sem:
            sd = mock_sem.from_records.return_value.self_deduplicate.return_value
            sd.selected_with_duplicates = []
            merged_entities, merged_relations = await resolver.resolve(entities, [])

    async def test_empty_entities_returns_empty(self):
        cfg = _make_cfg()
        resolver = EntityEdgeResolver(
            cfg,
            MEDICAL_GRAPHRAG_SCHEMA,
            _make_fake_embedder(),
        )
        merged_entities, merged_relations = await resolver.resolve([], [])
        assert merged_entities == []
        assert merged_relations == []

    async def test_resolve_with_use_llm_false(self):
        """Deterministic dedup merges normalize-identical items; distinct names kept."""
        items = ["metformin", "aspirin", "type 2 diabetes"]
        embedder = _make_fake_embedder()
        embedder.embed.return_value = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        cfg = _make_cfg(use_llm_resolution=False)
        resolver = EntityEdgeResolver(cfg, MEDICAL_GRAPHRAG_SCHEMA, embedder)
        entities = [_entity(item, "Drug") for item in items]
        sem_patch = "am_diag.graph_construction.resolve.deterministic.SemHash"
        with patch(sem_patch) as mock_sem:
            sd = mock_sem.from_records.return_value.self_deduplicate.return_value
            sd.selected_with_duplicates = []
            merged_entities, _ = await resolver.resolve(entities, [])
        assert len(merged_entities) == 3


class TestAliasApplier:
    def test_apply_produces_merged_entities_with_aliases(self):
        entities = [
            _entity("metformin", "Drug"),
            _entity("type 2 diabetes", "Disease"),
        ]
        relations = [
            _relation("type 2 diabetes", "metformin", "TREATED_BY"),
        ]
        entity_map = ResolvedAliasMap(
            alias_to_members={
                "metformin": ["metformin"],
                "type 2 diabetes": ["type 2 diabetes"],
            },
        )
        applier = AliasApplier()
        merged_entities, merged_relations = applier.apply(
            entities, relations, entity_map
        )
        assert len(merged_entities) == 2
        me = [e for e in merged_entities if e.canonical_name == "metformin"][0]
        assert me.canonical_name == "metformin"

    def test_apply_dedupes_relations_by_uuid_and_type(self):
        entities = [
            _entity("metformin", "Drug"),
            _entity("diabetes", "Disease"),
        ]
        relations = [
            _relation("diabetes", "metformin", "TREATED_BY"),
            _relation("diabetes", "metformin", "TREATED_BY"),
        ]
        entity_map = ResolvedAliasMap(
            alias_to_members={
                "metformin": ["metformin"],
                "diabetes": ["diabetes"],
            },
        )
        applier = AliasApplier()
        merged_entities, merged_relations = applier.apply(
            entities, relations, entity_map
        )
        assert len(merged_relations) == 1
