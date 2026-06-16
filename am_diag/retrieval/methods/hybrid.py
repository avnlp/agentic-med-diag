"""Atomic hybrid search method: store-native BM25 + dense vector fusion.

Delegates to the vector store's ``hybrid_search`` primitive (Weaviate primary
backend, Qdrant with sparse-vector support as a secondary option).
"""

from __future__ import annotations

from am_diag.common.data_models.vector_record import VectorHit
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.filters import SearchFilters
from am_diag.vector.embedding.base import Embedder


async def hybrid_search(
    store: VectorStoreBase,
    collection: str,
    query_text: str,
    embedder: Embedder,
    limit: int = 10,
    score_threshold: float | None = None,
    filters: SearchFilters | None = None,
    alpha: float = 0.5,
) -> list[VectorHit]:
    """Search a collection by store-native hybrid BM25 + dense vector.

    ``alpha`` controls the dense/sparse blend: 1.0 = pure dense,
    0.0 = pure sparse (BM25-style keyword).

    Args:
        store: Vector store instance.
        collection: Collection to search.
        query_text: Raw query text (for sparse leg).
        embedder: Embedder for the dense leg.
        limit: Maximum results.
        score_threshold: Minimum score threshold (applied client-side).
        filters: Optional metadata filters.
        alpha: Dense weight in [0, 1].

    Returns:
        Hits ordered by descending fused score.
    """
    query_vector = (await embedder.embed([query_text]))[0]
    payload_filter = filters.to_payload_filter() if filters else None
    hits = await store.hybrid_search(
        collection,
        query_vector,
        query_text,
        limit=limit,
        filters=payload_filter,
        alpha=alpha,
    )
    if score_threshold is not None:
        hits = [h for h in hits if h.score >= score_threshold]
    return hits


__all__ = ["hybrid_search"]
