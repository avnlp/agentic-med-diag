"""Chunk retriever: hybrid search on chunks, build results from payload.

Builds each ``Chunk`` straight from ``VectorHit.payload`` (no graph hydration).
When BFS is requested, seeds are the entities the chunks mention, fetched via
``chunk_seed_entities.cypher``.
"""

from __future__ import annotations

from typing import Any

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.retrieval.retrievers.base import HybridRetriever


class ChunkRetriever(HybridRetriever):
    """Hybrid search on the chunk collection, payload-built results."""

    name = "chunk"
    hydrate_cypher = None  # build straight from payload

    def _build(self, hit: VectorHit, row: dict[str, Any]) -> Chunk:
        """Build a ``Chunk`` from the hit payload."""
        p = hit.payload
        return Chunk(
            id=hit.id,
            text=p.get("text", ""),
            document_id=p.get("document_id", ""),
            document_source=p.get("document_source", ""),
            chunk_index=p.get("chunk_index", 0),
            chunk_size=p.get("chunk_size", len(p.get("text", ""))),
            cut_type=p.get("cut_type", "hybrid"),
        )

    async def _seed_ids(
        self,
        hits: list[VectorHit],
        rows: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Seed BFS from the entities the matched chunks mention."""
        ids = [str(h.id) for h in hits]
        out = await self._graph.execute_query(
            load("retrieval/chunk_seed_entities"),
            {"ids": ids},
        )
        return [r["id"] for r in out if r.get("id")]


__all__ = ["ChunkRetriever"]
