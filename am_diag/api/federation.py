"""Cross-graph (federated) retrieval over a project's graphs.

Each graph runs its own :class:`SearchEngine` (its own schema and embedder) and
yields a fused, ``source_graph_id``-tagged list. The per-graph lists are fused
once more across graphs — cross-encoder over the union when a reranker is
available, else RRF over the per-graph ranked lists. Text-based fusion is
schema-agnostic, so heterogeneous-schema graphs fuse correctly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.config import SearchConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.rerank.cross_encoder import cross_encoder_rerank
from am_diag.retrieval.rerank.rrf import rrf
from am_diag.vector.reranking.base import Reranker


if TYPE_CHECKING:
    from am_diag.api.graph import Graph


def _resolve_graphs(graphs: list[Graph]) -> list[Graph]:
    """Return a shallow copy of live graphs for concurrent fan-out."""
    return list(graphs)


async def federated_search(
    graphs: list[Graph],
    query: str,
    *,
    recipe: SearchConfig | str = "hybrid_rrf",
    filters: SearchFilters | None = None,
    limit: int = 20,
    rrf_k: int = 60,
    reranker: Reranker | None = None,
) -> list[SearchResult]:
    """Fan out per-graph search concurrently, then fuse across graphs.

    Args:
        graphs: Live graphs to search.
        query: Natural-language query.
        recipe: Recipe name or ``SearchConfig`` applied per graph.
        filters: Optional metadata filters.
        limit: Maximum fused results returned.
        rrf_k: RRF rank constant used when no reranker is given.
        reranker: Optional cross-encoder for cross-graph fusion.

    Returns:
        Fused, ranked results, each tagged with its ``source_graph_id``.
    """
    targets = _resolve_graphs(graphs)
    if not targets:
        return []
    per_graph = await asyncio.gather(
        *(graph.search(query, recipe=recipe, filters=filters) for graph in targets)
    )
    if reranker is not None:
        flat = [result for sublist in per_graph for result in sublist]
        return await cross_encoder_rerank(query, flat, reranker, top_k=limit)
    return rrf(list(per_graph), k=rrf_k)[:limit]


class FederatedSearch:
    """Adapter exposing ``.search(query, recipe)`` over a fixed graph set.

    The existing agent retrieval tools call ``.search`` on a single object, so
    wrapping a project's graphs in this adapter federates the agent unchanged.

    Args:
        client: The owning client.
        graph_ids: Graphs this adapter searches.
        limit: Maximum fused results returned per call.
    """

    def __init__(self, graphs: list[Graph], *, limit: int = 20) -> None:
        """Bind the adapter to a client and a fixed set of graphs."""
        self._graphs = list(graphs)
        self._limit = limit

    async def search(
        self,
        query: str,
        recipe: SearchConfig | str = "hybrid_rrf",
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Run a federated search over the bound graph set."""
        return await federated_search(
            self._graphs,
            query,
            recipe=recipe,
            filters=filters,
            limit=self._limit,
        )
