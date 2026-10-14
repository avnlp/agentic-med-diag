"""Community retriever: hybrid search on community reports, payload-built.

Builds each ``CommunityReport`` straight from ``VectorHit.payload`` (no graph
hydration). When BFS is requested, seeds are the member entities, fetched via
``community_seed_entities.cypher``. ``retrieve_all_at_level`` fetches every
report at a hierarchy level.
"""

from __future__ import annotations

from typing import Any

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.search import SearchResult
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.retrieval.retrievers.base import HybridRetriever


class CommunityRetriever(HybridRetriever):
    """Hybrid search on the community report collection, payload-built results."""

    name = "community"
    hydrate_cypher = None  # build straight from payload

    def _build(self, hit: VectorHit, row: dict[str, Any]) -> CommunityReport:
        """Build a ``CommunityReport`` from the hit payload."""
        p = hit.payload
        return CommunityReport(
            id=hit.id,
            community_id=p.get("community_id", ""),
            title=p.get("title", ""),
            summary=p.get("summary", ""),
            report_type=p.get("report_type", "full"),
        )

    async def _seed_ids(
        self,
        hits: list[VectorHit],
        rows: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Seed BFS from the member entities of the matched communities."""
        ids = [str(h.id) for h in hits]
        out = await self._graph.execute_query(
            load("retrieval/community_seed_entities"),
            {"ids": ids},
        )
        return [r["id"] for r in out if r.get("id")]

    async def retrieve_all_at_level(
        self,
        level: int | None = None,
    ) -> list[SearchResult]:
        """Fetch all community reports at a given hierarchy level.

        Args:
            level: Community hierarchy level. Defaults to config value.

        Returns:
            All community reports at that level, unsorted.
        """
        lvl = level if level is not None else self._config.community_report_level
        rows = await self._graph.execute_query(
            load("retrieval/reports_at_level"),
            {"level": lvl},
        )
        return [
            SearchResult(
                item=CommunityReport(
                    id=r["id"],
                    community_id=r["id"],
                    title=r.get("title", ""),
                    summary=r.get("summary", ""),
                ),
                score=1.0,
                source="community_all",
            )
            for r in rows
        ]


__all__ = ["CommunityRetriever"]
