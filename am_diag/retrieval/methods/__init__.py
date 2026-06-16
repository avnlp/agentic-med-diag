"""Atomic search methods for the retrieval layer.

Each method wraps a single backend primitive (vector search, fulltext search,
hybrid search, or BFS graph traversal) and returns ``list[SearchResult]``.
"""

from __future__ import annotations

from am_diag.retrieval.methods.bfs import bfs_expand
from am_diag.retrieval.methods.fulltext import fulltext_search
from am_diag.retrieval.methods.hybrid import hybrid_search
from am_diag.retrieval.methods.vector import vector_search


__all__ = [
    "bfs_expand",
    "fulltext_search",
    "hybrid_search",
    "vector_search",
]
