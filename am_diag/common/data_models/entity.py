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

    # Canonical external identifier for structured-graph entities, e.g.
    # "MONDO:0005148". Deliberately not a Dedup field: identity stays
    # name-keyed for text-extracted entities, while structured loaders pass an
    # explicit UUID5(curie) id. Kept first-class (not in schema_properties) so it
    # is a queryable Neo4j property.
    curie: str | None = None

    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)

    # Optional: text extraction sets this; structured-graph loaders leave it None.
    provenance: Provenance | None = None
