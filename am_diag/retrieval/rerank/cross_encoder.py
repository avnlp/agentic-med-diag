"""Cross-encoder reranker wrapper.

Uses the vector/reranking module's ``Reranker`` protocol to re-score results.
"""

from __future__ import annotations

from am_diag.common.data_models.search import SearchResult
from am_diag.vector.reranking.base import Reranker


async def cross_encoder_rerank(
    query: str,
    results: list[SearchResult],
    reranker: Reranker,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Re-rank ``results`` using a cross-encoder reranker.

    Args:
        query: The search query.
        results: Candidate results to re-score.
        reranker: Configured cross-encoder reranker instance.
        top_k: If set, only the top-``k`` results are returned.

    Returns:
        Results ordered by descending cross-encoder score.
    """
    if not results:
        return []

    # Build document text list from results
    doc_texts = [_item_to_text(r.item) for r in results]

    reranked = await reranker.rerank(query, doc_texts, top_k=top_k)

    # Map back to SearchResult
    out: list[SearchResult] = []
    for r in reranked:
        if r.index < len(results):
            orig = results[r.index]
            orig.score = r.score
            out.append(orig)

    return out


def _item_to_text(item: object) -> str:
    """Extract a text representation from a SearchResult item."""
    if isinstance(item, dict):
        return str(item.get("name", item.get("text", str(item))))
    if hasattr(item, "name") and item.name:
        return f"{item.__class__.__name__}: {item.name}"
    if hasattr(item, "text") and item.text:
        return str(item.text)
    return str(item)


__all__ = ["cross_encoder_rerank"]
