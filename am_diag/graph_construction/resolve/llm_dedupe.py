"""LLM-based deduplication per cluster via BAML IdentifyClusterDuplicates."""

from __future__ import annotations

import asyncio
import collections
from typing import Any

from am_diag.graph_construction.config import ExtractionPipelineConfig


async def _call_dedup(
    item: str,
    item_kind: str,
    candidates: list[str],
) -> Any:
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    if not candidates:
        return _identity_verdict(item)
    try:
        return await b.IdentifyClusterDuplicates(item, item_kind, candidates)
    except Exception:
        return _identity_verdict(item)


def _identity_verdict(item: str) -> Any:
    return collections.namedtuple("_Verdict", ["duplicates", "alias"])([], item)


class LLMDeduplicator:
    """LLM-based deduplication within a single cluster.

    For each item, retrieves top-*k* candidates via BM25+cosine fusion,
    then calls BAML ``IdentifyClusterDuplicates`` to find duplicates.
    """

    def __init__(self, config: ExtractionPipelineConfig) -> None:
        """Initialize deduplicator with pipeline config."""
        self._config = config

    async def dedupe_cluster(
        self,
        index_pool: list[int],
        items: list[str],
        embeddings: Any,
        bm25: Any,
        item_kind: str,
        llm_sem: asyncio.Semaphore,
    ) -> dict[str, list[str]]:
        """Deduplicate items in a single cluster using LLM.

        Args:
            index_pool: Indices into *items* for this cluster.
            items: All item strings.
            embeddings: ``(n, d)`` embedding array.
            bm25: Pre-built BM25 index.
            item_kind: ``"entity"`` or ``"edge"``.
            llm_sem: Concurrency limiter for LLM calls.

        Returns:
            Mapping of canonical alias → member strings.
        """
        from am_diag.graph_construction.resolve.candidates import top_k  # noqa: PLC0415

        pool = list(index_pool)
        alias_to_members: dict[str, list[str]] = {}
        bm25_w = self._config.resolution_bm25_weight
        emb_w = self._config.resolution_embedding_weight

        while pool:
            idx = pool.pop(0)
            item = items[idx]
            candidate_idxs = top_k(
                idx,
                items,
                embeddings,
                bm25,
                bm25_weight=bm25_w,
                embedding_weight=emb_w,
                k=self._config.resolution_top_k,
            )
            candidate_idxs = [i for i in candidate_idxs if i in pool]
            candidates = [items[i] for i in candidate_idxs]

            try:
                async with llm_sem:
                    verdict = await _call_dedup(item, item_kind, candidates)
            except Exception:
                verdict = _identity_verdict(item)

            duplicates_in_pool = [
                i for i in candidate_idxs if items[i] in verdict.duplicates
            ]
            members = [item] + [items[i] for i in duplicates_in_pool]
            for i in duplicates_in_pool:
                pool.remove(i)
            alias_to_members[verdict.alias] = members

        return alias_to_members


__all__ = ["LLMDeduplicator"]
