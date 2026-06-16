"""Relation — a structured SPO triple between two Entity nodes."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import Field

from am_diag.common.data_models.annotations import Embeddable
from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.provenance import Provenance


class Relation(DataPoint):
    """A directed relation between two entities.

    head_id/tail_id are None before entity resolution; set to UUID at resolution.
    No Dedup annotation — identity is uuid4 until resolved (relation embeddings
    keyed by DataPoint.id in the vector store).
    """

    head_id: UUID | None = None
    tail_id: UUID | None = None
    head_name: str = ""
    tail_name: str = ""
    type: str
    description: Annotated[str | None, Embeddable("Relation fact for search")] = None
    score: float = 1.0
    schema_properties: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
