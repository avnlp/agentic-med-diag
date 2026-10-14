"""Report models for the ingestion pipeline.

These summarize ingestion progress and success counts. ``IngestionReport`` is
returned by the ``Graph.add`` inline write path; ``ExtractionReport`` is produced
by the extraction pipeline nodes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractionReport(BaseModel):
    """Summary of a single extraction pipeline run on one batch of documents."""

    batch_id: str = ""
    chunk_count: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    entities_after_dedup: int = 0
    relations_after_dedup: int = 0
    entities_after_resolution: int = 0
    relations_after_resolution: int = 0
    communities_detected: int = 0
    reports_generated: int = 0
    output_path: str = ""


class IngestionReport(BaseModel):
    """Summary of a full ingestion pipeline run (extraction + DB write)."""

    batch_id: str = ""
    documents_ingested: int = 0
    chunks_ingested: int = 0
    entities_ingested: int = 0
    relations_ingested: int = 0
    communities_ingested: int = 0
    reports_ingested: int = 0
    embedding_collections_populated: list[str] = Field(default_factory=list)


__all__ = ["ExtractionReport", "IngestionReport"]
