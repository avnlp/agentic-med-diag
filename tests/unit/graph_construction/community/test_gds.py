"""Tests for whole-graph community detection (``GdsCommunityDetector.detect_all``).

``detect_all`` must run the Leiden projection without requiring in-memory
relations, while ``detect`` keeps its empty-relations skip guard. Neo4j is stubbed
at the ``execute_query`` boundary; no GDS plugin is exercised.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from am_diag.graph_construction.community.gds import GdsCommunityDetector


class _FakeGraph:
    """Minimal graph-store stub recording ``execute_query`` calls."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append((query, params or {}))
        return self._rows


class TestGdsDetectAll:
    """Whole-graph detection and the empty-relations guard."""

    async def test_detect_all_runs_leiden_without_relations(self) -> None:
        """detect_all projects/runs Leiden even with no relations in memory."""
        rows = [
            {
                "community_id": "c1",
                "community_level": 0,
                "entity_id": str(uuid4()),
                "parent_community_id": None,
            }
        ]
        graph = _FakeGraph(rows)
        communities = await GdsCommunityDetector(graph).detect_all()  # type: ignore[arg-type]
        assert len(graph.calls) == 1
        assert len(communities) == 1

    async def test_detect_skips_when_no_relations(self) -> None:
        """Detect keeps its guard: empty relations means no Leiden run."""
        graph = _FakeGraph([])
        result = await GdsCommunityDetector(graph).detect([])  # type: ignore[arg-type]
        assert result == []
        assert graph.calls == []
