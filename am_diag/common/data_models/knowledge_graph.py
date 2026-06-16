"""KnowledgeGraph — assembled result of a graph construction pipeline run."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema


class KnowledgeGraph(BaseModel):
    """Assembled output of graph construction (extraction → resolution → detection).

    Components do NOT take this as input — they work with explicit typed lists.
    This container is assembled by ingestion and consumed by storage and retrieval.
    """

    documents: list[Document] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    communities: list[Community] = Field(default_factory=list)
    reports: list[CommunityReport] = Field(default_factory=list)
    graph_schema: GraphSchema | None = None

    def entities_by_id(self) -> dict[UUID, Entity]:
        """Return a mapping from entity UUID to Entity."""
        return {e.id: e for e in self.entities}

    def chunks_by_id(self) -> dict[UUID, Chunk]:
        """Return a mapping from chunk UUID to Chunk."""
        return {c.id: c for c in self.chunks}

    def counts(self) -> dict[str, int]:
        """Return a summary of record counts per collection."""
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "communities": len(self.communities),
            "reports": len(self.reports),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize this KnowledgeGraph to a plain dict."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KnowledgeGraph":
        """Deserialize a KnowledgeGraph from a plain dict."""
        return cls.model_validate(d)
