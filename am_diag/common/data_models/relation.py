"""Relation — a structured SPO triple between two Entity nodes."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import Field

from am_diag.common.data_models.annotations import Dedup, Embeddable
from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.provenance import Provenance


class Relation(DataPoint):
    """A directed relation between two entities.

    Identity is a deterministic ``uuid5`` over the ``(head_id, tail_id, type)``
    Dedup keys, mirroring :class:`Entity`. Because ``head_id``/``tail_id`` are
    ``None`` until the deterministic endpoint-link step runs, a relation built
    straight from an extractor still gets a ``uuid4`` id; the id becomes stable
    only once endpoints are set and the relation is reconstructed
    (``ComposedExtractor._link_endpoints``). Structured-graph loaders set
    endpoints at construction, so their relations are deterministic immediately.
    """

    head_id: Annotated[UUID | None, Dedup()] = None
    tail_id: Annotated[UUID | None, Dedup()] = None
    head_name: str = ""
    tail_name: str = ""
    type: Annotated[str, Dedup()]
    description: Annotated[str | None, Embeddable("Relation fact for search")] = None
    score: float = 1.0
    schema_properties: dict[str, Any] = Field(default_factory=dict)

    # Structural edge label from a source graph schema, e.g. "ANA-GEN". `type`
    # carries the semantic relation (e.g. "EXPRESSED_IN"); this carries the
    # source's coarse endpoint-pair label. None for text-extracted relations.
    structural_label: str | None = None
    # Originating databases for structured edges, e.g. ["OPEN_TARGETS",
    # "DRUG_BANK"]. Empty for text-extracted relations.
    sources: list[str] = Field(default_factory=list)

    # Optional: text extraction sets this; structured-graph loaders leave it None.
    provenance: Provenance | None = None
