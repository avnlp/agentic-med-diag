"""Maximal Marginal Relevance (MMR) reranker.

Selects results that are both relevant to the query and diverse among
themselves. Useful when you have many similar results and want to avoid
redundancy.
"""

from __future__ import annotations

import hashlib

import numpy as np

from am_diag.common.data_models.search import SearchResult


def maximal_marginal_relevance(
    query_vec: list[float],
    results: list[SearchResult],
    lambda_: float = 0.7,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Re-rank results by MMR: balance relevance and diversity.

    Formula: ``MMR = lambda * rel(q, d) - (1-lambda) * max_sim(d, selected)``

    Args:
        query_vec: Query embedding vector.
        results: Candidate results (must have a ``score`` field).
        lambda_: Relevance weight in [0, 1]. Higher = more relevance-focused.
        top_k: Number of results to return (default: all).

    Returns:
        Results ordered by descending MMR score.
    """
    if not results:
        return []

    q = np.array(query_vec, dtype=np.float64)
    # Build similarity matrix from item vectors (approximate from scores)
    vecs = np.array(
        [_item_vector(r, len(query_vec)) for r in results],
        dtype=np.float64,
    )

    # Normalise vectors
    q = q / (np.linalg.norm(q) + 1e-10)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

    # Relevance = cosine similarity to query (use existing scores as proxy)
    relevance = np.array([r.score for r in results], dtype=np.float64)
    relevance = relevance / (np.max(relevance) + 1e-10)

    # Similarity matrix between items
    sim_matrix = vecs @ vecs.T

    selected: list[int] = []
    remaining = list(range(len(results)))

    for _ in range(top_k or len(results)):
        if not remaining:
            break
        mmr_scores = []
        for i in remaining:
            rel = relevance[i]
            max_sim = max(sim_matrix[i][j] for j in selected) if selected else 0.0
            mmr = lambda_ * rel - (1 - lambda_) * max_sim
            mmr_scores.append(mmr)

        best_idx = remaining[int(np.argmax(mmr_scores))]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [results[i] for i in selected]


def _item_vector(result: SearchResult, dim: int) -> list[float]:
    """Create a mock vector from result score for diversity computation.

    In production, the actual embedding should be looked up from the store.
    This fallback distributes the score across dimensions.
    """
    key = str(id(result.item))
    # Deterministic RNG seeding, not security
    seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)  # nosec
    rng = np.random.default_rng(seed)
    vec = rng.normal(0, 1, dim).astype(np.float64)
    vec = vec * (result.score / np.linalg.norm(vec) + 1e-10)
    return vec.tolist()


__all__ = ["maximal_marginal_relevance"]
