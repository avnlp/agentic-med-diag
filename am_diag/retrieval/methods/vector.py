"""Atomic dense vector search: embed query  search store."""

from __future__ import annotations

from am_diag.common.data_models.search import SearchResult
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.filters import SearchFilters
from am_diag.vector.embedding.base import Embedder


async def vector_search(
    store: VectorStoreBase,
    collection: str,
    query_text: str,
    embedder: Embedder,
    limit: int = 10,
    score_threshold: float | None = None,
    filters: SearchFilters | None = None,
) -> list[VectorHit]:
    """Search a collection by dense vector similarity.

    Embeds query_text with query-mode embedder and delegates to store.search.
    Returns store-level VectorHit objects (not yet hydrated to SearchResult).
    """
    query_vector = (await embedder.embed([query_text]))[0]
    payload_filter = filters.to_payload_filter() if filters else None
    return await store.search(
        collection,
        query_vector,
        limit=limit,
        score_threshold=score_threshold,
        filters=payload_filter,
    )


def hydrate_search_results(
    hits: list[VectorHit],
    source: str,
    matched_on: str | None = None,
    store: VectorStoreBase | None = None,
    collection: str | None = None,
) -> list[SearchResult]:
    """Convert raw VectorHit results to SearchResult items.

    Currently hydrates from payload data. When ``store`` and ``collection``
    are provided, full DataPoint objects could be fetched from the graph store.
    """
    results: list[SearchResult] = []
    for hit in hits:
        results.append(
            SearchResult(
                item=hit.payload,
                score=hit.score,
                source=source,
                matched_on=matched_on,
            ),
        )
    return results


__all__ = ["hydrate_search_results", "vector_search"]
