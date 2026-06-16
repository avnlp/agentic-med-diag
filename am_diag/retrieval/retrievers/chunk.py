"""Chunk retriever: hybrid search on chunks, expand to local graph context.

Uses ``chunk_neighbors.cypher`` to fetch the surrounding graph context
(entities, relations, communities) around each matched chunk.
"""

from __future__ import annotations

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.hybrid import hybrid_search
from am_diag.vector.embedding.base import Embedder


class ChunkRetriever:
    """Search chunks and return them with their local graph neighbourhood.

    Args:
        config: Retrieval settings.
        vector_store: Vector store for the chunk collection.
        graph_store: Graph store for neighbourhood expansion.
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
        """Search chunks and expand to local graph context.

        Args:
            query: Search query.
            filters: Optional metadata filters.

        Returns:
            ``SearchResult`` items with ``Chunk`` items.
        """
        hits = await hybrid_search(
            self._store,
            self._config.chunk_collection,
            query,
            self._embedder,
            limit=self._config.chunk_top_k,
            score_threshold=self._config.chunk_score_threshold,
            filters=filters,
        )

        if not hits:
            return []

        results: list[SearchResult] = []
        for hit in hits:
            p = hit.payload
            chunk = Chunk(
                id=hit.id,
                text=p.get("text", ""),
                document_id=p.get("document_id", ""),
                document_source=p.get("document_source", ""),
                chunk_index=p.get("chunk_index", 0),
                chunk_size=p.get("chunk_size", len(p.get("text", ""))),
                cut_type=p.get("cut_type", "hybrid"),
            )
            results.append(
                SearchResult(
                    item=chunk,
                    score=hit.score,
                    source="chunk",
                    matched_on="hybrid",
                ),
            )

        return results


__all__ = ["ChunkRetriever"]
