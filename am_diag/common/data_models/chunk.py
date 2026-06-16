"""Chunk data structure."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import Field

from am_diag.common.data_models.annotations import Embeddable
from am_diag.common.data_models.base import DataPoint


class Chunk(DataPoint):
    """A `Chunk` represents a text segment extracted from document.

    Carries full provenance back to its source `Document` via
    `document_id` and `document_source`.  The `id` field (inherited
    from `DataPoint`) is a deterministic UUID5 derived from
    ``(document_id, chunk_index)``.

    Attributes:
        text: The chunk's textual content.  Marked `Embeddable` for
            automatic vector indexing.
        document_id: UUID of the source `Document`.
        document_source: Corpus source (denormalized for fast filtering).
        chunk_index: Zero-based position of this chunk within the document.
        chunk_size: Character count of `text`.
        cut_type: How the chunk was split (``"sentence"``, ``"token"``,
            ``"recursive"``).
        properties: Arbitrary chunk metadata (strategy parameters,
            source-specific fields, etc.).
        title: Optional section or chapter heading.
    """

    text: Annotated[str, Embeddable("Chunk text for semantic search")]

    document_id: UUID
    document_source: str

    chunk_index: int
    chunk_size: int
    cut_type: str

    properties: dict[str, Any] = Field(default_factory=dict)
    title: str | None = None

    # Identity: same (document_id, chunk_index) → same UUID5
    metadata: dict[str, Any] = {
        "index_fields": ["text"],
        "identity_fields": ["document_id", "chunk_index"],
    }
