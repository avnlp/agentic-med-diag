"""Entity retriever: hybrid search on entities, hydrate via Cypher.

Hydrates each ``VectorHit`` into a full ``Entity`` via ``hydrate_entities.cypher``.
BFS expansion (when requested) seeds directly from the entity hits.
"""

from __future__ import annotations

from typing import Any

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.db.graph.serialize import reconstruct_schema_properties
from am_diag.retrieval.retrievers.base import HybridRetriever


class EntityRetriever(HybridRetriever):
    """Hybrid search on the entity collection with graph hydration."""

    name = "entity"
    hydrate_cypher = "retrieval/hydrate_entities"

    def _build(self, hit: VectorHit, row: dict[str, Any]) -> Entity:
        """Build a full ``Entity`` from the hydrated row, falling back to payload."""
        props = row.get("props", {}) or {}
        merged = {**hit.payload, **props}
        provenance = merged.get("provenance")
        if isinstance(provenance, str):
            provenance = (
                Provenance.model_validate_json(provenance) if provenance else None
            )
        return Entity(
            id=hit.id,
            name=merged.get("name", ""),
            label=merged.get("label", ""),
            description=merged.get("description"),
            score=merged.get("score", 1.0),
            curie=merged.get("curie"),
            canonical_name=merged.get("canonical_name"),
            aliases=merged.get("aliases") or [],
            provenance=provenance,
            schema_properties=reconstruct_schema_properties(Entity, props),
        )

    async def _seed_ids(
        self,
        hits: list[VectorHit],
        rows: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Seed BFS from the entity hits themselves."""
        return [str(h.id) for h in hits]


__all__ = ["EntityRetriever"]
