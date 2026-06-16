"""Reranker/fuser implementations.

Each module provides a reranking or fusion function that takes a list of
``SearchResult`` and returns a re-ranked list.
"""

from __future__ import annotations

from am_diag.retrieval.rerank.cross_encoder import cross_encoder_rerank
from am_diag.retrieval.rerank.mmr import maximal_marginal_relevance
from am_diag.retrieval.rerank.node_distance import node_distance_reranker
from am_diag.retrieval.rerank.rrf import rrf


__all__ = [
    "cross_encoder_rerank",
    "maximal_marginal_relevance",
    "node_distance_reranker",
    "rrf",
]
