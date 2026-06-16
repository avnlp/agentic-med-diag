"""Community — a clustered group of related medical entities."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field

from am_diag.common.data_models.annotations import Dedup, Embeddable
from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.provenance import Provenance


class Community(DataPoint):
    """A community of entities detected by graph partitioning.

    Identity = (community_id,). community_id is set by the detector (e.g. "0:42").
    """

    community_id: Annotated[str, Dedup("Community partition identifier")]
    level: int = 0
    parent: UUID | None = None
    title: str = ""
    summary: Annotated[str, Embeddable("Community summary for search")] = ""
    entity_ids: list[UUID] = Field(default_factory=list)
    relation_ids: list[UUID] = Field(default_factory=list)
    score: float = 1.0
    provenance: Provenance = Field(default_factory=Provenance)
