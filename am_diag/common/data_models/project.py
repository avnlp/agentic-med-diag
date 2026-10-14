"""Persisted descriptor of a Project (owned, shareable graph container)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class ProjectRecord(BaseModel):
    """Persisted descriptor of a Project.

    A Project is the level above a Graph: an owned, shareable container of graphs
    and the agent's federation boundary. Membership is M:N — a graph may belong
    to several projects.

    Attributes:
        project_id: Caller-chosen unique id.
        name: Optional display name.
        description: Optional display description.
        graph_ids: Member graphs (M:N — a graph may be in several projects).
        agent_settings: Optional per-project default ``AgentSettings`` overrides,
            stored as a dict and merged over ``AGENT_`` env defaults at run time.
        owner_id: Reserved for the future user model.
        members: Reserved for the future user model.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
    """

    project_id: str
    name: str | None = None
    description: str | None = None
    graph_ids: list[str] = Field(default_factory=list)
    agent_settings: dict[str, Any] | None = None
    owner_id: str | None = None
    members: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
