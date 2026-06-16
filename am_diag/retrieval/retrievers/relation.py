"""Relation retriever: hybrid search on relations, hydrate endpoint entities.

Uses ``hydrate_relations.cypher`` to fetch full relation edges (head/tail
entities, type, description) from the graph store.
"""

from __future__ import annotations

from uuid import UUID

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.hybrid import hybrid_search
from am_diag.vector.embedding.base import Embedder


class RelationRetriever:
    """Hybrid search on the ``relation`` collection with graph hydration.

    Args:
        config: Retrieval settings.
        vector_store: Vector store for the relation collection.
        graph_store: Graph store for hydration.
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
        """Search relations by semantic similarity to the query.

        Args:
            query: Natural-language query (e.g., "metformin treats diabetes").
            filters: Optional metadata filters.

        Returns:
            Hydrated ``SearchResult`` items, one per matched relation.
        """
        hits = await hybrid_search(
            self._store,
            self._config.relation_collection,
            query,
            self._embedder,
            limit=self._config.relation_top_k,
            score_threshold=self._config.relation_score_threshold,
            filters=filters,
        )

        if not hits:
            return []

        # Hydrate via graph store
        hit_ids = [str(h.id) for h in hits]
        relation_rows = await self._graph.execute_query(
            load("retrieval/hydrate_relations"),
            {"ids": hit_ids},
        )
        relations_by_id = {r["id"]: r for r in relation_rows}

        results: list[SearchResult] = []
        for hit in hits:
            rid = str(hit.id)
            row = relations_by_id.get(rid, {})
            props = row.get("props", {}) or {}
            relation = Relation(
                id=UUID(rid),
                head_name=props.get("head_name", ""),
                tail_name=props.get("tail_name", ""),
                type=props.get("type", ""),
                description=props.get("description", None),
            )
            results.append(
                SearchResult(
                    item=relation,
                    score=hit.score,
                    source="relation",
                    matched_on="hybrid",
                ),
            )

        return results


__all__ = ["RelationRetriever"]
