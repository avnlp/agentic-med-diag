"""Extraction provenance tracking for graph data points."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from am_diag.common.data_models.enums import ExtractorName


class Provenance(BaseModel):
    """Tracks which extractors produced a data point and from which chunks."""

    extractor: ExtractorName | None = None
    sources: list[ExtractorName] = Field(default_factory=list)
    surface_forms: list[str] = Field(default_factory=list)
    chunk_ids: list[UUID] = Field(default_factory=list)
    chunk_texts: list[str] = Field(default_factory=list)
    start_char: int | None = None
    end_char: int | None = None
