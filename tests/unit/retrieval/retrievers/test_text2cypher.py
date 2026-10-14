"""Tests for Text2CypherRetriever (am_diag/retrieval/retrievers/text2cypher.py)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.retrievers.text2cypher import Text2CypherRetriever


_BAML_TARGET = "am_diag.llm.baml_client.b"


def _retriever(graph: MagicMock, **config: object) -> Text2CypherRetriever:
    return Text2CypherRetriever(
        RetrievalConfig(**config), graph, MEDICAL_GRAPHRAG_SCHEMA
    )


def _baml(*cyphers: str) -> MagicMock:
    baml = MagicMock()
    baml.GenerateCypherQuery = AsyncMock(
        side_effect=[SimpleNamespace(cypher=c) for c in cyphers]
    )
    return baml


@pytest.fixture
def graph() -> MagicMock:
    """Return a graph store whose queries succeed with one arbitrary row."""
    store = MagicMock(spec=GraphStore)
    store.execute_query = AsyncMock(return_value=[{"drug": "metformin", "n": 3}])
    return store


class TestText2CypherRetriever:
    """Tests for generated-Cypher retrieval."""

    async def test_runs_generated_query_as_read_only(self, graph):
        """Verify the generated query is validated and executed read-only."""
        with patch(_BAML_TARGET, _baml("MATCH (n:Entity) RETURN n.name AS name")):
            results = await _retriever(graph).retrieve("what treats diabetes")

        calls = graph.execute_query.call_args_list
        assert [c.kwargs.get("read_only") for c in calls] == [True, True]
        assert calls[0].args[0].startswith("EXPLAIN ")
        assert [r.item.values for r in results] == [{"drug": "metformin", "n": 3}]

    async def test_returns_empty_when_read_only_execution_fails(self, graph):
        """Verify a write rejected by the server yields no results, not a crash."""
        graph.execute_query.side_effect = [[], RuntimeError("write in read mode")]
        with patch(_BAML_TARGET, _baml("MATCH (n) DETACH DELETE n")):
            results = await _retriever(graph).retrieve("wipe everything")

        assert results == []

    async def test_returns_empty_when_disabled(self, graph):
        """Verify no query is generated when text2cypher is turned off."""
        results = await _retriever(graph, use_text_to_cypher=False).retrieve("q")

        assert results == []
        graph.execute_query.assert_not_called()
