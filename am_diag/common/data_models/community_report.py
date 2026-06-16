"""CommunityReport — an LLM-generated summary of a community."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from am_diag.common.data_models.annotations import Embeddable
from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.provenance import Provenance


class Finding(BaseModel):
    """A structured finding extracted from a community report."""

    summary: str
    explanation: str


class CommunityReport(DataPoint):
    """An LLM-generated summary of one graph community.

    No Dedup — each pipeline run produces a new report (uuid4). The community
    it references is identified by community_id.
    """

    community_id: UUID
    report_type: str = "full"
    title: str = ""
    summary: Annotated[str, Embeddable("Community report summary")] = ""
    findings: list[Finding] = Field(default_factory=list)
    rating: float = 0.0
    rating_explanation: str = ""
    rank: float = 0.0
    provenance: Provenance = Field(default_factory=Provenance)
