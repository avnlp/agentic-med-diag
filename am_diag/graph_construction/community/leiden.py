"""Leiden community detection via graspologic-native."""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

from am_diag.common.data_models.community import Community
from am_diag.common.data_models.relation import Relation
from am_diag.graph_construction.community.base import CommunityDetector, build_edge_list
from am_diag.graph_construction.config import CommunityDetectionConfig


logger = logging.getLogger(__name__)


class LeidenCommunityDetector(CommunityDetector):
    """Leiden clustering via graspologic-native.

    Pure component — no IO. Returns hierarchical ``Community`` DataPoints.
    """

    def __init__(self, config: CommunityDetectionConfig | None = None) -> None:
        """Initialize detector with optional config."""
        self._config = config or CommunityDetectionConfig()

    async def detect(self, relations: list[Relation]) -> list[Community]:
        """Run hierarchical Leiden on the relation graph.

        Args:
            relations: Resolved ``Relation`` DataPoints.

        Returns:
            List of ``Community`` DataPoints with ``entity_ids`` populated.
        """
        try:
            from graspologic_native import (  # noqa: PLC0415
                hierarchical_leiden,  # type: ignore
            )
        except ImportError:
            raise ImportError(
                "Leiden community detection requires:"
                " pip install 'am_diag[community_detection]'",
            ) from None

        edges = build_edge_list(relations)
        if not edges:
            logger.warning("No edges to cluster — skipping community detection")
            return []

        clusters = hierarchical_leiden(
            edges,
            max_cluster_size=self._config.max_cluster_size,
            resolution=self._config.resolution,
            randomness=self._config.randomness,
            use_modularity=self._config.use_modularity,
            iterations=self._config.iterations,
            seed=self._config.seed,
            starting_communities={},
        )

        agg: dict[tuple[int, str], dict] = {}
        for cluster in clusters:
            cid = f"{cluster.level}:{cluster.cluster}"
            key = (cluster.level, cid)
            if key not in agg:
                agg[key] = {
                    "community_id": cid,
                    "level": cluster.level,
                    "parent": None,
                    "entity_ids": [],
                }
            if cluster.parent_cluster is not None:
                agg[key]["parent"] = f"{cluster.level - 1}:{cluster.parent_cluster}"
            agg[key]["entity_ids"].append(cluster.node)

        communities: list[Community] = []
        for level_cid, data in agg.items():
            parent_id = data["parent"]
            parent_uuid: UUID | None = None
            if parent_id is not None:
                parent_key = (int(parent_id.split(":")[0]), parent_id)
                if parent_key in agg:
                    parent_uuid = uuid4()
            communities.append(
                Community(
                    community_id=f"{level_cid[0]}:{level_cid[1]}",
                    level=data["level"],
                    parent=parent_uuid,
                    entity_ids=[UUID(n) for n in data["entity_ids"]],
                ),
            )

        return communities


__all__ = ["LeidenCommunityDetector"]
