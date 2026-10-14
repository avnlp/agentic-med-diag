"""Tests for Graph retrieval, context packing, and the guarded query().

A real ``Graph`` is built with fake dependencies (stub SearchEngine and graph
store) and preloaded into the Workspace cache, so Graph's own logic is exercised
without touching a backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from am_diag.api.errors import SchemaError
from am_diag.api.graph import Graph
from am_diag.api.workspace import Workspace
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA


_COLLECTIONS = {
    "entity": "e",
    "relation": "r",
    "chunk": "c",
    "community": "k",
}


def _chunk(text: str) -> Chunk:
    """Build a valid Chunk with required provenance fields."""
    return Chunk(
        text=text,
        document_id=uuid4(),
        document_source="test",
        chunk_index=0,
        chunk_size=len(text),
        cut_type="recursive",
    )


def _client_with_fake_graph(tmp_path, search_results=None, execute_rows=None):
    """Return (client, graph) with a real Graph over fakes preloaded for 'med'."""
    client = Workspace.open(
        registry_path=tmp_path / "registry.json", state_dir=tmp_path / "state"
    )
    search_engine = SimpleNamespace(
        search=AsyncMock(return_value=list(search_results or []))
    )
    graph_store = SimpleNamespace(
        execute_query=AsyncMock(return_value=list(execute_rows or [])),
        close=AsyncMock(),
    )
    graph = Graph(
        graph_store=graph_store,
        vector_store=SimpleNamespace(close=AsyncMock()),
        embedder=SimpleNamespace(model_id="fake-model", close=AsyncMock()),
        search_engine=search_engine,
        schema=MEDICAL_GRAPHRAG_SCHEMA,
        collections=_COLLECTIONS,
        graph_id="med",
        state_dir=tmp_path / "state",
    )
    client.attach_graph(graph)
    return client, graph


class TestGraphSearch:
    """Tests for Graph.search and Graph.context."""

    async def test_search_tags_source_graph_id(self, tmp_path):
        """Verify every fused result is tagged with the graph id."""
        results = [
            SearchResult(
                item=Entity(name="sepsis", label="Disease"), score=0.9, source="entity"
            )
        ]
        client, _ = _client_with_fake_graph(tmp_path, search_results=results)
        out = await client.graph("med").search("sepsis")
        assert out[0].source_graph_id == "med"

    async def test_search_applies_limit(self, tmp_path):
        """Verify the post-fusion limit truncates the result list."""
        results = [
            SearchResult(
                item=Entity(name=f"e{i}", label="Disease"), score=1.0, source="entity"
            )
            for i in range(5)
        ]
        client, _ = _client_with_fake_graph(tmp_path, search_results=results)
        out = await client.graph("med").search("q", limit=2)
        assert len(out) == 2

    async def test_context_respects_budget(self, tmp_path):
        """Verify context output never exceeds the character budget."""
        results = [
            SearchResult(item=_chunk("x" * 50), score=1.0, source="chunk")
            for _ in range(10)
        ]
        client, _ = _client_with_fake_graph(tmp_path, search_results=results)
        ctx = await client.graph("med").context("q", max_characters=200)
        assert "x" * 50 in ctx
        assert len(ctx) <= 200


class TestGraphQuery:
    """Tests for the guarded read-only query()."""

    async def test_rejects_write_cypher(self, tmp_path):
        """Verify a write clause is rejected in read-only mode."""
        client, _ = _client_with_fake_graph(tmp_path)
        with pytest.raises(SchemaError):
            await client.graph("med").query("MATCH (n) CREATE (m) RETURN n")

    async def test_allows_read_cypher(self, tmp_path):
        """Verify a pure read query is forwarded to the store."""
        client, graph = _client_with_fake_graph(tmp_path, execute_rows=[{"n": 1}])
        out = await client.graph("med").query("MATCH (n) RETURN n LIMIT 1")
        assert out == [{"n": 1}]
        graph.graph_store.execute_query.assert_awaited_once()
        assert graph.graph_store.execute_query.await_args.kwargs["read_only"] is True

    async def test_write_allowed_when_not_read_only(self, tmp_path):
        """Verify write Cypher is forwarded when read_only is disabled."""
        client, graph = _client_with_fake_graph(tmp_path, execute_rows=[])
        await client.graph("med").query("CREATE (n)", read_only=False)
        graph.graph_store.execute_query.assert_awaited_once()
