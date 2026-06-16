"""Community detection base class and shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

from am_diag.common.data_models.community import Community
from am_diag.common.data_models.relation import Relation


def build_edge_list(
    relations: list[Relation],
) -> list[tuple[str, str, float]]:
    """Build deduplicated undirected edge list from relations.

    Returns ``(entity_uuid_a, entity_uuid_b, weight)`` tuples
    where ``a < b`` lexicographically.
    """
    seen: set[tuple[str, str]] = set()
    edges: list[tuple[str, str, float]] = []
    for rel in relations:
        if not rel.head_id or not rel.tail_id:
            continue
        u, v = sorted([str(rel.head_id), str(rel.tail_id)])
        if (u, v) in seen:
            continue
        seen.add((u, v))
        edges.append((u, v, rel.score if rel.score else 1.0))
    return edges


class CommunityDetector(ABC):
    """Abstract base for community detection backends.

    Subclasses implement ``detect`` to return hierarchical ``Community``
    DataPoints from resolved relations. The method is async to support
    backends that call external services (e.g. Neo4j GDS).
    """

    @abstractmethod
    async def detect(self, relations: list[Relation]) -> list[Community]:
        """Detect communities from a list of resolved relations.

        Args:
            relations: Resolved relations with ``head_id``/``tail_id``.

        Returns:
            List of ``Community`` DataPoints.
        """


__all__ = ["CommunityDetector", "build_edge_list"]
