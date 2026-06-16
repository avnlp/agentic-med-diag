"""Vector store boundary models: upsert records and search hits."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """Transport record for upserting one vector into a store collection.

    The collection name is NOT a field — it is the upsert call argument,
    allowing the same record type to target any of the four collections
    (entity, relation, chunk, community).
    """

    id: UUID
    vector: list[float] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorHit(BaseModel):
    """A single result returned from a vector store similarity search.

    Store-level result (before hydration into full DataPoint objects).
    The retrieval layer converts VectorHit → SearchResult by fetching the
    full DataPoint from the graph store using the id.
    """

    id: UUID
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)
