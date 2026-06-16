"""Unified retrieval result model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation


class SearchResult(BaseModel):
    """A single result from the retrieval layer, after hydration.

    Unifies the five stray result classes previously in retrieval/models.py.
    The `item` discriminated union allows consumers to inspect the concrete type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: Entity | Relation | Chunk | CommunityReport
    score: float
    source: str
    matched_on: str | None = None
