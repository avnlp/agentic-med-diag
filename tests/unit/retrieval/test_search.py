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


class TestResolveRecipe:
    """Tests for _resolve_recipe helper."""

    def test_known_recipe(self) -> None:
        """Verify a known recipe name resolves to the correct SearchConfig."""
        config = _resolve_recipe("entity")
        assert config.methods == ["entity"]

    def test_unknown_recipe_raises(self) -> None:
        """Verify an unknown recipe name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown recipe"):
            _resolve_recipe("nonexistent")

    def test_hybrid_rrf_resolves(self) -> None:
        """Verify hybrid_rrf recipe resolves to include all search methods."""
        config = _resolve_recipe("hybrid_rrf")
        assert "entity" in config.methods
        assert "community" in config.methods

    def test_text2cypher_resolves(self) -> None:
        """Verify text2cypher recipe resolves to the text2cypher method."""
        config = _resolve_recipe("text2cypher")
        assert config.methods == ["text2cypher"]


class TestSearchEngine:
    """Tests for SearchEngine — top-level recipe-driven retrieval."""

    @pytest.fixture
    def engine(self) -> SearchEngine:
        """Return a SearchEngine with mocked dependencies."""
        return SearchEngine(
            config=RetrievalConfig(),
            vector_store=MagicMock(spec=VectorStoreBase),
            graph_store=MagicMock(spec=GraphStore),
            embedder=MagicMock(spec=Embedder),
        )

    async def test_empty_methods_returns_empty(self, engine: SearchEngine) -> None:
        """Verify no search methods produce an empty result list."""
        results = await engine.search(
            "query",
            recipe=SearchConfig(methods=[], reranker="rrf"),
        )
        assert results == []

    async def test_search_with_string_recipe(self, engine: SearchEngine) -> None:
        """Verify searching with a string recipe name works correctly."""
        engine._entity.retrieve = AsyncMock(return_value=[])
        results = await engine.search("query", recipe="entity")
        assert results == []

    async def test_search_with_hybrid_rrf(self, engine: SearchEngine) -> None:
        """Verify hybrid_rrf recipe routes to all four retrievers."""
        for retriever in [
            engine._entity,
            engine._relation,
            engine._chunk,
            engine._community,
        ]:
            retriever.retrieve = AsyncMock(return_value=[])

        results = await engine.search("query", recipe="hybrid_rrf")
        assert results == []
        engine._entity.retrieve.assert_awaited_once()
        engine._relation.retrieve.assert_awaited_once()
        engine._chunk.retrieve.assert_awaited_once()
        engine._community.retrieve.assert_awaited_once()

    async def test_search_fuses_results(self, engine: SearchEngine) -> None:
        """Verify results from multiple retrievers are fused into a single list."""
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

    async def test_bfs_flags_forwarded_to_retrievers(
        self, engine: SearchEngine
    ) -> None:
        """Verify bfs and bfs_depth flags are passed to the retriever."""
        for retriever in [
            engine._entity,
            engine._relation,
            engine._chunk,
            engine._community,
        ]:
            retriever.retrieve = AsyncMock(return_value=[])

        recipe = SearchConfig(methods=["entity"], reranker="rrf", bfs=True, bfs_depth=3)
        await engine.search("q", recipe=recipe)

        engine._entity.retrieve.assert_awaited_once()
        kwargs = engine._entity.retrieve.await_args.kwargs
        assert kwargs["bfs"] is True
        assert kwargs["bfs_depth"] == 3

    async def test_filters_results_by_score(self, engine: SearchEngine) -> None:
        """Verify results below the reranker_min_score threshold are filtered out."""
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
