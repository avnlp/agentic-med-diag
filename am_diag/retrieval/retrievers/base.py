"""Shared template for hybrid-search retrievers.

``HybridRetriever`` owns the common flow — hybrid search, optional graph
hydration, ``SearchResult`` construction, and optional BFS expansion — behind a
single ``retrieve()`` method. Concrete retrievers declare ``name`` and
``hydrate_cypher`` and implement ``_build``; they override ``_seed_ids`` to
enable BFS expansion from their hits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.search import SearchResult
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.methods.bfs import bfs_expand
from am_diag.retrieval.methods.hybrid import hybrid_search
from am_diag.vector.embedding.base import Embedder


class HybridRetriever(ABC):
    """Template for hybrid-search retrievers over one vector collection.

    Subclasses declare:
        name: Config prefix and ``SearchResult.source`` label (e.g. ``"entity"``).
            Collection / top_k / score_threshold are resolved from
            ``RetrievalConfig`` by this name (``f"{name}_collection"`` etc.).
        hydrate_cypher: Retrieval Cypher key to fetch full rows, or ``None`` to
            build results straight from ``VectorHit.payload``.

    and implement ``_build``. Override ``_seed_ids`` to enable BFS expansion.

    Args:
        config: Retrieval settings.
        vector_store: Vector store for this retriever's collection.
        graph_store: Graph store for hydration and BFS.
        embedder: Embedder for query vectorisation.
    """

    name: str
    hydrate_cypher: str | None = None

    def __init__(
        self,
        config: RetrievalConfig,
        vector_store: VectorStoreBase,
        graph_store: GraphStore,
        embedder: Embedder,
    ) -> None:
        """Initialize retriever with config and dependencies."""
        self._config = config
        self._store = vector_store
        self._graph = graph_store
        self._embedder = embedder

    # ── public interface ────────────────────────────────────────────
    async def retrieve(
        self,
        query: str,
        filters: SearchFilters | None = None,
        bfs: bool = False,
        bfs_depth: int | None = None,
    ) -> list[SearchResult]:
        """Hybrid-search the collection, build results, optionally BFS-expand.

        Args:
            query: Search query.
            filters: Optional metadata filters.
            bfs: When True, expand from this retriever's seed entities via
                ``bfs_expand``.
            bfs_depth: Max hop depth for BFS (defaults to config traversal depth).

        Returns:
            Hydrated ``SearchResult`` items, plus any BFS neighbours.
        """
        hits = await hybrid_search(
            self._store,
            self._collection,
            query,
            self._embedder,
            limit=self._top_k,
            score_threshold=self._score_threshold,
            filters=filters,
            alpha=self._config.hybrid_alpha,
        )
        if not hits:
            return []

        rows = await self._hydrate(hits)
        results = [
            SearchResult(
                item=self._build(hit, rows.get(str(hit.id), {})),
                score=hit.score,
                source=self.name,
                matched_on="hybrid",
            )
            for hit in hits
        ]

        if bfs:
            seeds = await self._seed_ids(hits, rows)
            if seeds:
                results.extend(
                    await bfs_expand(
                        self._graph,
                        seeds,
                        depth=(
                            self._config.traversal_depth
                            if bfs_depth is None
                            else bfs_depth
                        ),
                        limit=self._config.traversal_limit,
                        filters=filters,
                    ),
                )
        return results

    # ── config resolved by naming convention ────────────────────────
    @property
    def _collection(self) -> str:
        """Vector collection name for this retriever (``f"{name}_collection"``)."""
        return getattr(self._config, f"{self.name}_collection")

    @property
    def _top_k(self) -> int:
        """Result limit for this retriever (``f"{name}_top_k"``)."""
        return getattr(self._config, f"{self.name}_top_k")

    @property
    def _score_threshold(self) -> float | None:
        """Optional score floor (``f"{name}_score_threshold"``, None if absent)."""
        return getattr(self._config, f"{self.name}_score_threshold", None)

    # ── shared hydration ────────────────────────────────────────────
    async def _hydrate(self, hits: list[VectorHit]) -> dict[str, dict[str, Any]]:
        """Fetch full rows keyed by id, or ``{}`` on the payload path."""
        if not self.hydrate_cypher:
            return {}
        rows = await self._graph.execute_query(
            load(self.hydrate_cypher),
            {"ids": [str(h.id) for h in hits]},
        )
        return {r["id"]: r for r in rows}

    # ── per-type seams ──────────────────────────────────────────────
    @abstractmethod
    def _build(self, hit: VectorHit, row: dict[str, Any]) -> Any:
        """Build the domain object for one hit.

        Args:
            hit: The raw vector hit (use ``hit.payload`` on the payload path).
            row: The hydrated row for this hit, or ``{}`` when ``hydrate_cypher``
                is None.

        Returns:
            The domain object to wrap in a ``SearchResult``.
        """

    async def _seed_ids(
        self,
        hits: list[VectorHit],
        rows: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Entity UUIDs to seed BFS expansion from.

        Default returns ``[]`` (no expansion). Override per retriever to map
        hits to the entity nodes BFS should traverse from.
        """
        return []


__all__ = ["HybridRetriever"]
