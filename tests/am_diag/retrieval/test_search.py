"""Tests for SearchEngine — top-level recipe-driven retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.config import RetrievalConfig, SearchConfig
from am_diag.retrieval.search import SearchEngine, _resolve_recipe
from am_diag.vector.embedding.base import Embedder


pytestmark = pytest.mark.enable_socket


class TestResolveRecipe:
    def test_known_recipe(self) -> None:
        config = _resolve_recipe("entity")
        assert config.methods == ["entity"]

    def test_unknown_recipe_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown recipe"):
            _resolve_recipe("nonexistent")

    def test_hybrid_rrf_resolves(self) -> None:
        config = _resolve_recipe("hybrid_rrf")
        assert "entity" in config.methods
        assert "community" in config.methods

    def test_text2cypher_resolves(self) -> None:
        config = _resolve_recipe("text2cypher")
        assert config.methods == ["text2cypher"]


class TestSearchEngine:
    @pytest.fixture
    def engine(self) -> SearchEngine:
        return SearchEngine(
            config=RetrievalConfig(),
            vector_store=MagicMock(spec=VectorStoreBase),
            graph_store=MagicMock(spec=GraphStore),
            embedder=MagicMock(spec=Embedder),
        )

    async def test_empty_methods_returns_empty(self, engine: SearchEngine) -> None:
        results = await engine.search(
            "query",
            recipe=SearchConfig(methods=[], reranker="rrf"),
        )
        assert results == []

    async def test_search_with_string_recipe(self, engine: SearchEngine) -> None:
        engine._entity.retrieve = AsyncMock(return_value=[])
        results = await engine.search("query", recipe="entity")
        assert results == []

    async def test_search_with_hybrid_rrf(self, engine: SearchEngine) -> None:
        for retriever in [
            engine._entity,
            engine._relation,
            engine._chunk,
            engine._community,
        ]:
            retriever.retrieve = AsyncMock(return_value=[])

        results = await engine.search("query", recipe="hybrid_rrf")
        assert results == []

    async def test_search_fuses_results(self, engine: SearchEngine) -> None:
        entity = Entity(id=uuid4(), name="metformin", label="Drug")
        engine._entity.retrieve = AsyncMock(
            return_value=[SearchResult(item=entity, score=0.9, source="entity")],
        )
        engine._relation.retrieve = AsyncMock(return_value=[])
        engine._chunk.retrieve = AsyncMock(return_value=[])
        engine._community.retrieve = AsyncMock(return_value=[])

        results = await engine.search("metformin", recipe="hybrid_rrf")
        assert len(results) > 0
        assert results[0].source == "entity"

    async def test_filters_results_by_score(self, engine: SearchEngine) -> None:
        entity = Entity(id=uuid4(), name="test", label="Drug")
        engine._entity.retrieve = AsyncMock(
            return_value=[SearchResult(item=entity, score=0.05, source="entity")],
        )
        engine._relation.retrieve = AsyncMock(return_value=[])
        engine._chunk.retrieve = AsyncMock(return_value=[])
        engine._community.retrieve = AsyncMock(return_value=[])
        engine._config.reranker_min_score = 0.1

        results = await engine.search("test", recipe="hybrid_rrf")
        assert len(results) == 0
