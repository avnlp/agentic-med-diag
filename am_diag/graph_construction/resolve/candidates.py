"""BM25 + cosine candidate generation for LLM deduplication."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from am_diag.graph_construction.resolve.text import normalize_text


def _tokenize(items: list[str]) -> list[list[str]]:
    return [normalize_text(t).split() for t in items]


def build_bm25(items: list[str]) -> BM25Okapi:
    """Build a BM25 index from *items*."""
    return BM25Okapi(_tokenize(items))


def top_k(
    query_idx: int,
    items: list[str],
    embeddings: np.ndarray,
    bm25: BM25Okapi,
    bm25_weight: float = 0.5,
    embedding_weight: float = 0.5,
    k: int = 16,
) -> list[int]:
    """Return top-*k* candidate indices for *query_idx* via fused BM25+cosine.

    Args:
        query_idx: Index of the query item.
        items: All item strings.
        embeddings: ``(n, d)`` embedding array.
        bm25: Pre-built BM25 index.
        bm25_weight: BM25 score weight in [0, 1].
        embedding_weight: Cosine score weight in [0, 1].
        k: Number of candidates to return.

    Returns:
        Indices of top candidates, excluding the query itself.
    """
    tokenized = _tokenize([items[query_idx]])
    bm25_scores = np.asarray(bm25.get_scores(tokenized[0]))
    embedding_scores = cosine_similarity(
        embeddings[query_idx : query_idx + 1],
        embeddings,
    ).flatten()
    fused = bm25_weight * bm25_scores + embedding_weight * embedding_scores
    ranked = np.argsort(fused)[::-1]
    return [int(i) for i in ranked if i != query_idx][:k]


__all__ = ["build_bm25", "top_k"]
