"""Unified retrieval result model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation


class CypherRow(BaseModel):
    """One raw row returned by a generated Cypher query.

    Generated queries return arbitrary columns, so a row cannot be mapped onto
    ``Entity``, ``Relation``, ``Chunk`` or ``CommunityReport``.
    """

    values: dict[str, Any]

    def __str__(self) -> str:
        """Render the row as ``column: value`` pairs."""
        return ", ".join(f"{key}: {value}" for key, value in self.values.items())


class SearchResult(BaseModel):
    """A single result from the retrieval layer, after hydration.

    Unifies the five stray result classes previously in retrieval/models.py.
    The `item` discriminated union allows consumers to inspect the concrete type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: Entity | Relation | Chunk | CommunityReport | CypherRow
    score: float
    source: str
    matched_on: str | None = None

    # Provenance for federated (cross-graph) search. None for single-graph
    # results; set to the originating graph_id when a Project fuses graphs.
    source_graph_id: str | None = None
