"""Community retriever: hybrid search on community reports, expand to members.

Uses ``community_members.cypher`` to fetch entities associated with
each matched community.
"""

from __future__ import annotations

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.hybrid import hybrid_search
from am_diag.vector.embedding.base import Embedder


class CommunityRetriever:
    """Hybrid search on community report collection with member expansion.

    Args:
        config: Retrieval settings.
        vector_store: Vector store for the community collection.
        graph_store: Graph store for member expansion.
        embedder: Embedder for query vectorisation.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        vector_store: VectorStoreBase,
        graph_store: GraphStore,
        embedder: Embedder,
    ) -> None:
        """Initialize retriever with config and dependencies."""
        self._config = config
        self._store = vector_store
        self._graph = graph_store
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Search community reports and expand to member entities.

        Args:
            query: Search query.
            filters: Optional metadata filters.

        Returns:
            ``SearchResult`` items with ``CommunityReport`` items.
        """
        hits = await hybrid_search(
            self._store,
            self._config.community_collection,
            query,
            self._embedder,
            limit=self._config.community_top_k,
            filters=filters,
        )

        if not hits:
            return []

        results: list[SearchResult] = []
        for hit in hits:
            p = hit.payload
            report = CommunityReport(
                id=hit.id,
                community_id=p.get("community_id", ""),
                title=p.get("title", ""),
                summary=p.get("summary", ""),
                report_type=p.get("report_type", "full"),
            )
            results.append(
                SearchResult(
                    item=report,
                    score=hit.score,
                    source="community",
                    matched_on="hybrid",
                ),
            )

        return results

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

        await self._graph.execute_query(
            load("retrieval/community_members"),
            {"community_id": None},
        )
        _query = """
        MATCH (cr:CommunityReport)-[:HAS_REPORT]-(c:Community)
        WHERE c.level = $level
        RETURN cr.id AS id, cr.title AS title, cr.summary AS summary
        """
        report_rows = await self._graph.execute_query(_query, {"level": lvl})

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
            for r in report_rows
        ]


__all__ = ["CommunityRetriever"]
