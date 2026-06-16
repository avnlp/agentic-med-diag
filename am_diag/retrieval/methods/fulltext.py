"""Atomic fulltext (sparse/BM25) search method.

Delegates to the vector store's hybrid backend with ``alpha=0`` to obtain a
pure sparse/BM25 ranking. When the backend does not support ``hybrid_search``,
falls back to ``vector_search`` with an empty dense vector (store-dependent).
"""

from __future__ import annotations

from am_diag.common.data_models.vector_record import VectorHit
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.filters import SearchFilters


async def fulltext_search(
    store: VectorStoreBase,
    collection: str,
    query_text: str,
    limit: int = 10,
    filters: SearchFilters | None = None,
) -> list[VectorHit]:
    """Search a collection by BM25-style keyword matching.

    Uses the store's hybrid search with alpha=0 (pure sparse) if available.
    When ``hybrid_search`` is not supported by the backend, this raises
    ``NotImplementedError``.
    """
    # Use an empty query vector since alpha=0 means pure sparse
    payload_filter = filters.to_payload_filter() if filters else None
    try:
        return await store.hybrid_search(
            collection,
            query_vector=[0.0],
            query_text=query_text,
            limit=limit,
            filters=payload_filter,
            alpha=0.0,
        )
    except NotImplementedError:
        raise NotImplementedError(
            f"fulltext_search is not supported by {type(store).__name__}; "
            "the backend must implement _hybrid_search."
        ) from None


__all__ = ["fulltext_search"]
