"""Typed state for LangGraph ingestion pipelines.

The state flows through extraction → embedding → graph-write phases,
accumulating documents, chunks, entities, and relations.
"""

from __future__ import annotations

from typing import Any, TypedDict

from am_diag.common.data_models import (
    Chunk,
    Community,
    CommunityReport,
    Document,
    Entity,
    KnowledgeGraph,
    Relation,
    VectorRecord,
)
from am_diag.ingestion.models import ExtractionReport, IngestionReport


class IngestionState(TypedDict, total=False):
    """Accumulated state through the ingestion pipeline.

    Every field has a default or ``None`` so the graph can be initialised
    with just ``{"batch_id": "..."}``.
    """

    batch_id: str
    documents: list[Document]
    chunks: list[Chunk]
    entities: list[Entity]
    relations: list[Relation]
    communities: list[Community]
    reports: list[CommunityReport]
    kg: KnowledgeGraph
    extraction_report: ExtractionReport
    ingestion_report: IngestionReport
    output_path: str
    records: list[VectorRecord]
    embedding_collections_populated: list[str]
    config: dict[str, Any] | None


__all__ = ["IngestionState"]
