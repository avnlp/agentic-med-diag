"""Persisted descriptor of a Graph deployment in the Registry."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.schema import GraphSchema


_GRAPH_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _default_schema() -> GraphSchema:
    """Return a private copy of the bundled medical schema.

    The bundled schemas are imported on first use so that importing the data
    models does not load every schema module.
    """
    from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA  # noqa: PLC0415

    return MEDICAL_GRAPHRAG_SCHEMA.model_copy(deep=True)


class GraphRecord(BaseModel):
    """Persisted descriptor of a Graph deployment in the Registry.

    A Graph is a fully separate backend deployment: its own Neo4j instance, its
    own internal vector store, and its own ``GraphSchema``. The record holds only
    connection metadata (env-var names, never secrets) plus the schema, so it is
    safe to serialize to the on-disk registry.

    Attributes:
        graph_id: Caller-chosen unique id.
        name: Optional display name.
        description: Optional display description.
        neo4j: Neo4j connection metadata.
        vector: Internal vector-store connection metadata.
        graph_schema: This graph's schema, driving extraction and text2cypher.
        owner_id: Reserved for the future user model (nullable today).
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
    """

    graph_id: str
    name: str | None = None
    description: str | None = None
    neo4j: GraphConnection
    vector: VectorConnection
    graph_schema: GraphSchema = Field(default_factory=_default_schema)
    owner_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("graph_id")
    @classmethod
    def _validate_graph_id(cls, graph_id: str) -> str:
        """Require an id that is safe to use as a cache file name."""
        if ".." in graph_id or _GRAPH_ID_RE.fullmatch(graph_id) is None:
            raise ValueError(
                "graph_id may contain only letters, digits, '.', '_' and '-', "
                "and must start with a letter or digit"
            )
        return graph_id
