"""Tests for whole-graph finalization (community detection).

``finalize_graph`` must run whole-graph detection once and upsert + link the
result. Structured-graph and corpus ingestion are now covered by the durable
job engine tests. External boundaries (graph store, detector) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from am_diag.common.data_models import Community
from am_diag.ingestion import graph_ingestion as gi


class TestFinalizeGraph:
    """Graph-global community detection, persistence, and linking."""

    async def test_detects_and_links_communities(self, monkeypatch) -> None:
        """finalize_graph runs detect_all and upserts + links the communities."""
        graph_store = AsyncMock()
        communities = [Community(community_id="0:1", entity_ids=[])]
        detector = AsyncMock()
        detector.detect_all = AsyncMock(return_value=communities)
        monkeypatch.setattr(
            "am_diag.graph_construction.community.gds.GdsCommunityDetector",
            lambda graph, config: detector,
        )

        count = await gi.finalize_graph(graph_store)

        assert count == 1
        graph_store.upsert_communities.assert_awaited_once_with(communities)
        graph_store.link_in_community.assert_awaited_once_with(communities)
        graph_store.link_parent_community.assert_awaited_once_with(communities)
