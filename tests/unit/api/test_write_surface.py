"""Tests for the Graph write surface (ingest / add / add_fact).

A real Graph over mocked store/vector/embedder is preloaded into the Workspace
cache with a real temp content cache, so the write paths run offline. The
structured (entities/relations) path is used for ``add`` so no extraction model
is built.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from am_diag.api.graph import Graph
from am_diag.api.registry import Registry
from am_diag.api.workspace import Workspace
from am_diag.common.data_models.entity import Entity
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.ingestion.cache import CacheManager
from am_diag.ingestion.models import IngestionReport


_COLLECTIONS = {"entity": "e", "relation": "r", "chunk": "c", "community": "k"}


def _setup(tmp_path):
    """Preload a real Graph over fakes for 'med'; return (client, graph, live_jobs).

    The workspace and the prebuilt graph share one ``live_jobs`` registry and
    cache manager (injected), so a job created via the graph is visible to the
    workspace without reaching into private attributes.
    """
    registry = Registry.open(tmp_path / "registry.json")
    live_jobs: dict = {}
    cache_manager = CacheManager(tmp_path / "cache")
    client = Workspace(
        registry,
        state_dir=tmp_path / "state",
        cache_manager=cache_manager,
        live_jobs=live_jobs,
    )
    embedder = SimpleNamespace(
        embed=AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts]),
        model_id="fake-model",
    )
    graph = Graph(
        graph_store=SimpleNamespace(
            write_knowledge_graph=AsyncMock(), close=AsyncMock()
        ),
        vector_store=SimpleNamespace(upsert=AsyncMock(), close=AsyncMock()),
        embedder=embedder,
        search_engine=SimpleNamespace(),
        schema=MEDICAL_GRAPHRAG_SCHEMA,
        collections=_COLLECTIONS,
        graph_id="med",
        state_dir=tmp_path / "state",
        cache=cache_manager.for_graph("med"),
        job_registry=live_jobs,
    )
    client.attach_graph(graph)
    return client, graph, live_jobs


class TestWriteSurface:
    """Tests for ingest/add/add_fact."""

    async def test_ingest_returns_draft_job(self, tmp_path):
        """Verify ingest returns a non-running job registered as live."""
        client, _, live_jobs = _setup(tmp_path)
        job = await client.graph("med").ingest([])
        assert job.status == "draft"
        assert live_jobs[job.id] is job

    async def test_add_structured_returns_report(self, tmp_path):
        """Verify add() with entities runs the engine and returns a report."""
        client, graph, _ = _setup(tmp_path)
        report = await client.graph("med").add(
            entities=[Entity(name="sepsis", label="Disease")]
        )
        assert isinstance(report, IngestionReport)
        assert report.chunks_ingested == 1
        graph.graph_store.write_knowledge_graph.assert_awaited_once()

    async def test_add_fact_writes_single_relation(self, tmp_path):
        """Verify add_fact builds and writes exactly one relation."""
        client, graph, _ = _setup(tmp_path)
        rel = await client.graph("med").add_fact(
            head="aspirin", relation="TREATS", tail="fever"
        )
        assert rel.type == "TREATS"
        assert rel.head_name == "aspirin"
        graph.graph_store.write_knowledge_graph.assert_awaited_once()
        kg = graph.graph_store.write_knowledge_graph.call_args.args[0]
        assert len(kg.relations) == 1
        assert len(kg.entities) == 2
