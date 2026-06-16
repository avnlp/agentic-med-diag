"""Graph-aware node distance reranker.

Prefers results that are closer (in hops) to a given centre entity ID.
Useful when the retrieval already has BFS results and you want to
emphasise nearby nodes.
"""

from __future__ import annotations

from am_diag.common.data_models.search import SearchResult


def node_distance_reranker(
    center_id: str,
    results: list[SearchResult],
    decay: float = 0.5,
) -> list[SearchResult]:
    """Re-score results by proximity to ``center_id``.

    For results that have a ``matched_on`` field starting with ``hop_``,
    the score is boosted by ``decay ** hop_distance``. Other results
    keep their original score.

    Args:
        center_id: Entity UUID (string) to measure distance from.
        results: Candidate results.
        decay: Hop-distance decay factor (0 < decay < 1).

    Returns:
        Results re-ordered by proximity-weighted score.
    """
    results = list(results)  # shallow copy
    for r in results:
        matched = r.matched_on or ""
        if matched.startswith("hop_"):
            try:
                hop = int(matched.split("_")[1])
                r.score = r.score * (decay**hop)
            except (ValueError, IndexError):
                pass

    return sorted(results, key=lambda r: -r.score)


__all__ = ["node_distance_reranker"]
