"""Entity retriever: hybrid search on entities, optional BFS expansion.

Hydrates ``VectorHit`` -> ``Entity`` via ``hydrate_entities.cypher``,
then optionally expands to neighbours via ``bfs_expand``.
"""

from __future__ import annotations

from uuid import UUID

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.bfs import bfs_expand
from am_diag.retrieval.methods.hybrid import hybrid_search
from am_diag.vector.embedding.base import Embedder


class EntityRetriever:
    """Hybrid search on entity collection, optionally BFS-expand from hit IDs.

    Args:
        config: Retrieval settings.
        vector_store: Vector store for the entity collection.
        graph_store: Graph store for hydration and BFS.
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
        bfs: bool = False,
        bfs_depth: int | None = None,
    ) -> list[SearchResult]:
        """Search entities, hydrate, and optionally BFS-expand.

        Args:
            query: Search query.
            filters: Optional metadata filters.
            bfs: When True, expand from entity hits via ``bfs_expand``.
            bfs_depth: Max hop depth for BFS (default from config).

        Returns:
            Hydrated ``SearchResult`` items.
        """
        # 1. Hybrid search over entity collection
        hits = await hybrid_search(
            self._store,
            self._config.entity_collection,
            query,
            self._embedder,
            limit=self._config.entity_top_k,
            score_threshold=self._config.entity_score_threshold,
            filters=filters,
        )

        if not hits:
            return []

        # 2. Hydrate entities via graph store
        hit_ids = [str(h.id) for h in hits]
        entity_rows = await self._graph.execute_query(
            load("retrieval/hydrate_entities"),
            {"ids": hit_ids},
        )
        entities_by_id = {r["id"]: r for r in entity_rows}

        # 3. Build SearchResults
        results: list[SearchResult] = []
        for hit in hits:
            eid = str(hit.id)
            row = entities_by_id.get(eid, {})
            props = row.get("props", {}) or {}
            entity = Entity(
                id=UUID(eid),
                name=props.get("name", hit.payload.get("name", "")),
                label=props.get("label", hit.payload.get("label", "")),
            )
            results.append(
                SearchResult(
                    item=entity,
                    score=hit.score,
                    source="entity",
                    matched_on="hybrid",
                ),
            )

        # 4. Optional BFS expansion
        if bfs and hit_ids:
            bfs_results = await bfs_expand(
                self._graph,
                hit_ids,
                depth=bfs_depth or self._config.traversal_depth,
                limit=self._config.traversal_limit,
                filters=filters,
            )
            results.extend(bfs_results)

        return results


__all__ = ["EntityRetriever"]
