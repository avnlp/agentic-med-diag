"""Top-level ``search()`` entry point.

Fans out over the recipe's named methods concurrently, applies the requested
reranker, and returns fused ``SearchResult`` items.
"""

from __future__ import annotations

import asyncio
from typing import Any

from am_diag.common.data_models.schema import GraphSchema
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval import recipes
from am_diag.retrieval.config import RetrievalConfig, SearchConfig
from am_diag.retrieval.filters import SearchFilters
from am_diag.retrieval.rerank import cross_encoder_rerank, rrf
from am_diag.retrieval.retrievers.chunk import ChunkRetriever
from am_diag.retrieval.retrievers.community import CommunityRetriever
from am_diag.retrieval.retrievers.entity import EntityRetriever
from am_diag.retrieval.retrievers.relation import RelationRetriever
from am_diag.retrieval.retrievers.text2cypher import Text2CypherRetriever
from am_diag.vector.embedding.base import Embedder
from am_diag.vector.reranking.base import Reranker


class SearchEngine:
    """Orchestrates multi-strategy retrieval with recipe-driven fan-out.

    Constructed once with all dependencies; call ``search()`` per query.

    Args:
        config: Retrieval configuration.
        vector_store: Vector store (all four collections).
        graph_store: Graph database client.
        embedder: Embedder for query vectorisation.
        schema: Optional GraphSchema for Text2Cypher retriever.
        reranker: Optional cross-encoder reranker (needed for
            ``HYBRID_CROSS_ENCODER`` recipe).
    """

    def __init__(
        self,
        config: RetrievalConfig,
        vector_store: VectorStoreBase,
        graph_store: GraphStore,
        embedder: Embedder,
        schema: GraphSchema | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize search engine with config and dependencies."""
        self._config = config
        self._reranker = reranker

        # Build retrievers once
        self._entity = EntityRetriever(config, vector_store, graph_store, embedder)
        self._relation = RelationRetriever(config, vector_store, graph_store, embedder)
        self._chunk = ChunkRetriever(config, vector_store, graph_store, embedder)
        self._community = CommunityRetriever(
            config,
            vector_store,
            graph_store,
            embedder,
        )
        self._text2cypher: Any = (
            Text2CypherRetriever(config, graph_store, schema) if schema else None
        )

    async def search(
        self,
        query: str,
        recipe: SearchConfig | str = "hybrid_rrf",
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Execute ``recipe`` against ``query`` and return fused results.

        Args:
            query: Natural-language query.
            recipe: ``SearchConfig`` instance or recipe name (key in
                ``am_diag.retrieval.recipes``).
            filters: Optional metadata filters.

        Returns:
            Ranked ``SearchResult`` items, truncated to ``recipe.limit``.
        """
        if isinstance(recipe, str):
            recipe = _resolve_recipe(recipe)

        if not recipe.methods:
            return []

        raw_results = await self._fan_out(query, recipe, filters)
        return await self._fuse(query, recipe, raw_results)

    async def _fan_out(
        self,
        query: str,
        recipe: SearchConfig,
        filters: SearchFilters | None,
    ) -> list[list[SearchResult]]:
        """Fan out over recipe methods concurrently."""
        method_map: dict[str, Any] = {
            "entity": self._entity.retrieve,
            "relation": self._relation.retrieve,
            "chunk": self._chunk.retrieve,
            "community": self._community.retrieve,
        }
        if self._text2cypher:
            method_map["text2cypher"] = self._text2cypher.retrieve

        tasks = []
        for method in recipe.methods:
            fn = method_map.get(method)
            if fn is None:
                continue
            if method == "entity":
                tasks.append(
                    fn(
                        query,
                        filters=filters,
                        bfs=recipe.bfs,
                        bfs_depth=recipe.bfs_depth,
                    )
                )
            else:
                tasks.append(fn(query, filters=filters))

        return await asyncio.gather(*tasks)

    async def _fuse(
        self,
        query: str,
        recipe: SearchConfig,
        raw_results: list[list[SearchResult]],
    ) -> list[SearchResult]:
        """Fuse results using the recipe's reranker strategy."""
        non_empty = [r for r in raw_results if r]
        if not non_empty:
            return []

        if recipe.reranker == "cross_encoder" and self._reranker:
            flat = [item for sublist in non_empty for item in sublist]
            fused = await cross_encoder_rerank(query, flat, self._reranker)
        else:
            fused = rrf(non_empty, k=self._config.rrf_k)

        if self._config.reranker_min_score > 0:
            fused = [r for r in fused if r.score >= self._config.reranker_min_score]

        return fused[: recipe.limit]


def _resolve_recipe(name: str) -> SearchConfig:
    """Look up a recipe by name."""
    recipe_map = {
        "entity": recipes.ENTITY,
        "relation": recipes.RELATION,
        "chunk": recipes.CHUNK,
        "community": recipes.COMMUNITY,
        "hybrid_rrf": recipes.HYBRID_RRF,
        "hybrid_cross_encoder": recipes.HYBRID_CROSS_ENCODER,
        "bfs_expand": recipes.BFS_EXPAND,
        "text2cypher": recipes.TEXT2CYPHER,
    }
    recipe = recipe_map.get(name)
    if recipe is None:
        raise ValueError(f"Unknown recipe: {name!r}; options: {list(recipe_map)}")
    return recipe


__all__ = ["SearchEngine"]
