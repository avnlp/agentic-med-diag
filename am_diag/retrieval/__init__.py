"""Retrieval layer: layered multi-strategy search with recipe-driven orchestration.

Top-level usage::

    engine = SearchEngine(config, vector_store, graph_store, embedder)
    results = await engine.search("diabetes metformin", recipe="hybrid_rrf")
"""

from __future__ import annotations

from am_diag.retrieval import recipes
from am_diag.retrieval.config import RetrievalConfig, SearchConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.search import SearchEngine


__all__ = [
    "RetrievalConfig",
    "SearchConfig",
    "SearchEngine",
    "SearchFilters",
    "recipes",
]
