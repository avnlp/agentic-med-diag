"""Entity — a medical named entity tracked through the extraction pipeline."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from am_diag.common.data_models.annotations import Dedup, Embeddable, LLMContext
from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.provenance import Provenance


class Entity(DataPoint):
    """A medical entity: disease, drug, symptom, gene, etc.

    Identity = (name, label). name+label are Dedup keys for UUID5 generation.
    """

    name: Annotated[str, Embeddable("Primary entity name"), Dedup()]
    label: Annotated[str, Dedup()]
    description: Annotated[str | None, LLMContext()] = None
    score: float = 1.0
    schema_properties: dict[str, Any] = Field(default_factory=dict)

    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)

    provenance: Provenance = Field(default_factory=Provenance)
