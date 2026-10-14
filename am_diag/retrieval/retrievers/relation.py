"""Relation retriever: hybrid search on relations, hydrate endpoint entities.

Uses ``hydrate_relations.cypher`` to fetch full relation edges (props plus the
``head_id`` / ``tail_id`` endpoints). BFS expansion seeds from those endpoints.
"""

from __future__ import annotations

from typing import Any

from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.db.graph.serialize import reconstruct_schema_properties
from am_diag.retrieval.retrievers.base import HybridRetriever


class RelationRetriever(HybridRetriever):
    """Hybrid search on the ``relation`` collection with graph hydration."""

    name = "relation"
    hydrate_cypher = "retrieval/hydrate_relations"

    def _build(self, hit: VectorHit, row: dict[str, Any]) -> Relation:
        """Build a ``Relation`` from the hydrated row props."""
        props = row.get("props", {}) or {}
        return Relation(
            id=hit.id,
            head_id=row.get("head_id"),
            tail_id=row.get("tail_id"),
            head_name=props.get("head_name", ""),
            tail_name=props.get("tail_name", ""),
            type=props.get("type", ""),
            description=props.get("description"),
            structural_label=props.get("structural_label"),
            sources=props.get("sources", []),
            schema_properties=reconstruct_schema_properties(Relation, props),
        )

    async def _seed_ids(
        self,
        hits: list[VectorHit],
        rows: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Seed BFS from each relation's head and tail entity ids."""
        seeds: list[str] = []
        for row in rows.values():
            for key in ("head_id", "tail_id"):
                val = row.get(key)
                if val:
                    seeds.append(str(val))
        return seeds


__all__ = ["RelationRetriever"]
