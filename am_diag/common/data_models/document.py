"""Document — the primary unit of knowledge corpus text."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from am_diag.common.data_models.annotations import Dedup, Embeddable
from am_diag.common.data_models.base import DataPoint


class Document(DataPoint):
    """A single document from a knowledge corpus.

    Identity = (source, external_id). If a loader has no stable external_id,
    it should set external_id to a content hash so the document UUID stays stable
    across runs.
    """

    text: Annotated[str, Embeddable("Document text")]
    source: Annotated[str, Dedup()]
    external_id: Annotated[str | None, Dedup()] = None
    title: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
