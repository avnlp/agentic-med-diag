"""Tests for retrieval recipe tools."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from am_diag.agents.tools import _serialize_results, make_retrieval_tools
from am_diag.common.data_models import SearchResult
from am_diag.common.data_models.entity import Entity
from am_diag.retrieval.search import SearchEngine


class TestSerializeResults:
    def test_entity_serialization(self) -> None:
        entity = Entity(name="metformin", label="Drug")
        result = SearchResult(item=entity, score=0.95, source="test")
        serialized = _serialize_results([result])
        assert len(serialized) == 1
        assert serialized[0]["score"] == 0.95
        assert serialized[0]["source"] == "test"
        assert serialized[0]["item"]["name"] == "metformin"

    def test_empty_list(self) -> None:
        assert _serialize_results([]) == []


class TestMakeRetrievalTools:
    def test_returns_seven_tools(self) -> None:
        engine = MagicMock(spec=SearchEngine)
        engine.search = AsyncMock(return_value=[])
        tools = make_retrieval_tools(engine)
        assert len(tools) == 7

    def test_all_tools_have_names(self) -> None:
        engine = MagicMock(spec=SearchEngine)
        engine.search = AsyncMock(return_value=[])
        tools = make_retrieval_tools(engine)
        names = {t.name for t in tools}
        assert "entity_search" in names
        assert "relation_search" in names
        assert "chunk_search" in names
        assert "community_search" in names
        assert "hybrid_search" in names
        assert "cypher_search" in names
        assert "global_map_reduce" in names

    def test_entity_search_calls_engine(self) -> None:
        engine = MagicMock(spec=SearchEngine)
        engine.search = AsyncMock(return_value=[])
        tools = make_retrieval_tools(engine)
        entity_search = [t for t in tools if t.name == "entity_search"][0]
        asyncio.run(entity_search.ainvoke({"query": "diabetes"}))
        engine.search.assert_called()
