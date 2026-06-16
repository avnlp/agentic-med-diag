"""Neo4j GDS community detection (Leiden via GDS procedures).

Runs Leiden via Neo4j Graph Data Science (GDS) library procedures:
project the ``Entity``-``:RELATES_TO`` graph, run ``gds.leiden.mutate``
with ``includeIntermediateCommunities=true``, and read hierarchical
community assignments back as ``Community`` DataPoints.

Requires the GDS plugin to be installed on the Neo4j instance.
Zero additional LLM calls beyond the Leiden computation.
"""

from __future__ import annotations

import logging
import uuid
from uuid import UUID

from am_diag.common.data_models.community import Community
from am_diag.common.data_models.relation import Relation
from am_diag.db.graph.base import GraphStore
from am_diag.graph_construction.community.base import CommunityDetector
from am_diag.graph_construction.config import CommunityDetectionConfig


logger = logging.getLogger(__name__)


class GdsCommunityDetector(CommunityDetector):
    """Neo4j GDS Leiden community detection.

    Projects the entity graph in Neo4j, runs GDS Leiden, and reads
    hierarchical community assignments. The ``detect`` method is async
    because it issues Cypher queries via the graph store.

    Args:
        graph_store: Connected Neo4j client.
        config: Community detection parameters (``seed`` forwarded to GDS).
    """

    def __init__(
        self,
        graph_store: GraphStore,
        config: CommunityDetectionConfig | None = None,
    ) -> None:
        """Initialize GDS detector with graph store and config."""
        self._graph = graph_store
        self._config = config or CommunityDetectionConfig()

    @staticmethod
    def _collect_entity_ids(relations: list[Relation]) -> list[str]:
        """Extract unique entity UUID strings from resolved relations."""
        ids: list[str] = []
        for rel in relations:
            if rel.head_id:
                ids.append(str(rel.head_id))
            if rel.tail_id:
                ids.append(str(rel.tail_id))
        return list(set(ids))

    @staticmethod
    def _aggregate_rows(
        rows: list[dict],
    ) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], str | None]]:
        """Aggregate GDS result rows into communities_by_key and parent_map."""
        communities_by_key: dict[tuple[str, int], dict] = {}
        parent_map: dict[tuple[str, int], str | None] = {}

        for row in rows:
            cid = row.get("community_id", "")
            level = int(row.get("community_level", 0))
            entity_id = row.get("entity_id", "")
            parent_cid = row.get("parent_community_id")

            if not cid or not entity_id:
                continue

            key = (cid, level)
            if key not in communities_by_key:
                communities_by_key[key] = {
                    "community_id": f"{level}:{cid}",
                    "level": level,
                    "entity_ids": [],
                }
                parent_val = (
                    str(parent_cid) if parent_cid and parent_cid != "null" else None
                )
                parent_map[key] = parent_val

            try:
                communities_by_key[key]["entity_ids"].append(UUID(entity_id))
            except (ValueError, AttributeError):
                continue

        return communities_by_key, parent_map

    @staticmethod
    def _build_communities(
        communities_by_key: dict[tuple[str, int], dict],
        parent_map: dict[tuple[str, int], str | None],
    ) -> list[Community]:
        """Convert aggregated data into Community DataPoints."""
        communities: list[Community] = []
        id_cache: dict[str, UUID] = {}

        for key, data in communities_by_key.items():
            comm_id_str = data["community_id"]
            if comm_id_str not in id_cache:
                id_cache[comm_id_str] = uuid.uuid4()
            comm_uuid = id_cache[comm_id_str]

            parent_uuid: UUID | None = None
            parent_val = parent_map.get(key)
            if parent_val:
                parent_key_str = f"{data['level'] - 1}:{parent_val}"
                if parent_key_str not in id_cache:
                    id_cache[parent_key_str] = uuid.uuid4()
                parent_uuid = id_cache[parent_key_str]

            communities.append(
                Community(
                    id=comm_uuid,
                    community_id=comm_id_str,
                    level=data["level"],
                    parent=parent_uuid,
                    entity_ids=data["entity_ids"],
                ),
            )

        return communities

    async def detect(self, relations: list[Relation]) -> list[Community]:
        """Run GDS Leiden via Neo4j and return hierarchical Community Datapoints.

        The Cypher template at ``community/run_gds_leiden`` handles graph
        projection, Leiden execution, result reading, and cleanup. Each result
        row is converted to a ``Community`` DataPoint with ``entity_ids``
        listing the UUIDs of entities assigned to that community.

        Args:
            relations: Resolved relations (used to extract entity IDs for the
                graph projection scope). Pass an empty list to detect
                communities over all entities in the graph.

        Returns:
            List of ``Community`` DataPoints, one per assigned entity per
            hierarchical level. Empty if GDS is unavailable or the
            execution fails.
        """
        from am_diag.common.cypher.loader import load  # noqa: PLC0415

        entity_ids = self._collect_entity_ids(relations)
        if not entity_ids:
            logger.warning("GDS: no entity IDs found in relations, skipping")
            return []

        graph_name = f"gds_leiden_{uuid.uuid4().hex[:8]}"
        cypher = load("community/run_gds_leiden")

        try:
            rows = await self._graph.execute_query(
                cypher,
                {
                    "graph_name": graph_name,
                    "seed": self._config.seed,
                },
            )
        except Exception as exc:
            logger.warning(
                "GDS Leiden failed (is the GDS plugin installed?): %s",
                exc,
            )
            return []

        if not rows:
            logger.info("GDS Leiden produced no community assignments")
            return []

        communities_by_key, parent_map = self._aggregate_rows(rows)
        communities = self._build_communities(communities_by_key, parent_map)

        n_levels = len({c.level for c in communities})
        logger.info(
            "GDS Leiden detected %d communities across %d levels",
            len(communities),
            n_levels,
        )
        return communities


__all__ = ["GdsCommunityDetector"]
