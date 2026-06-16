"""EntityEdgeResolver — two-stage embedding-cluster + LLM entity/edge resolution."""

from __future__ import annotations

import asyncio
from typing import Literal

import numpy as np

from am_diag.common.data_models.cluster import ItemCluster, ResolvedAliasMap
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.resolve.apply import AliasApplier
from am_diag.graph_construction.resolve.candidates import build_bm25
from am_diag.graph_construction.resolve.cluster import cluster_items
from am_diag.graph_construction.resolve.deterministic import DeterministicDeduplicator
from am_diag.graph_construction.resolve.llm_dedupe import LLMDeduplicator
from am_diag.vector.embedding.base import Embedder


ItemKind = Literal["entity", "edge"]


class EntityEdgeResolver:
    """Two-stage embedding-cluster + LLM resolution of entities/edges.

    Stage 1 (no LLM): embed, deterministic pre-filter, cluster.
    Stage 2 (BAML LLM): per-cluster LLM deduplication with BM25+cosine
    candidate retrieval. Symmetric for entities and edge types.
    """

    def __init__(
        self,
        config: ExtractionPipelineConfig,
        schema: GraphSchema,
        embedder: Embedder,
    ) -> None:
        """Initialize resolver with config, schema, and embedder."""
        self._config = config
        self._schema = schema
        self._embedder = embedder

    async def resolve(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> tuple[list[Entity], list[Relation]]:
        """Resolve duplicate aggregated entities/relations.

        Mutates DataPoints in place:
        entities get ``canonical_name``, ``aliases``;
        relations get ``head_id``/``tail_id``.

        Args:
            entities: Aggregated-stage entities.
            relations: Aggregated-stage relations.

        Returns:
            ``(merged_entities, merged_relations)``.
        """
        if not entities:
            return [], []

        entity_texts = [e.name for e in entities]
        entity_alias_map = await self._resolve_items(entity_texts, "entity")
        edge_alias_map = await self._resolve_items(
            sorted({r.type for r in relations}),
            "edge",
        )

        applier = AliasApplier()
        return applier.apply(entities, relations, entity_alias_map, edge_alias_map)

    async def _resolve_items(
        self,
        items: list[str],
        item_kind: ItemKind,
    ) -> ResolvedAliasMap:
        if not items:
            return ResolvedAliasMap()

        # Stage 1: deterministic pre-filter
        deduper = DeterministicDeduplicator(self._config)
        pre_map = deduper.dedupe(items, item_kind)
        residual = [k for k, v in pre_map.alias_to_members.items() if len(v) > 0]

        if not self._config.use_llm_resolution or len(residual) <= 1:
            return pre_map

        # Stage 2: cluster + LLM dedupe
        embeddings = np.asarray(
            await self._embedder.embed(residual),
            dtype=np.float32,
        )
        clusters = await asyncio.to_thread(
            cluster_items,
            embeddings,
            self._config.resolution_cluster_size,
            self._config.resolution_kmeans_max_iter,
        )
        bm25 = await asyncio.to_thread(build_bm25, residual)

        llm_deduper = LLMDeduplicator(self._config)
        llm_sem = asyncio.Semaphore(self._config.resolution_max_concurrent_llm_calls)
        cluster_sem = asyncio.Semaphore(
            self._config.resolution_max_concurrent_clusters,
        )

        async def run_cluster(cluster: ItemCluster) -> dict[str, list[str]]:
            async with cluster_sem:
                return await llm_deduper.dedupe_cluster(
                    list(cluster.items),
                    residual,
                    embeddings,
                    bm25,
                    item_kind,
                    llm_sem,
                )

        llm_results = await asyncio.gather(
            *(run_cluster(c) for c in clusters),
        )
        merged: dict[str, list[str]] = {}
        for partial in llm_results:
            merged.update(partial)

        return ResolvedAliasMap(alias_to_members=merged)


__all__ = ["EntityEdgeResolver"]
