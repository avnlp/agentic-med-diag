"""GraphSchema: entity and relation type definitions for the medical knowledge graph."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from am_diag.common.data_models.annotations import Dedup
from am_diag.common.data_models.base import DataPoint


class EntityType(DataPoint):
    """A type definition for medical entities (config node, not extracted entity).

    Identity = (label,).
    """

    label: Annotated[str, Dedup()]
    gliner_label: str
    description: str
    key_property: str = "name"
    metadata: dict = {"index_fields": ["gliner_label"], "identity_fields": ["label"]}


class RelationType(DataPoint):
    """A type definition for medical relations (config node, not extracted relation).

    Identity = (type,).
    """

    type: Annotated[str, Dedup()]
    glirel_label: str
    head_labels: list[str] = Field(default_factory=list)
    tail_labels: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)
    metadata: dict = {"index_fields": ["glirel_label"], "identity_fields": ["type"]}


class GraphSchema(BaseModel):
    """Runtime-injectable KG schema — entity and relation type definitions.

    Used by extractors, resolvers, community summarizers, text2cypher, and agents.
    """

    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)

    def gliner_labels(self) -> list[str]:
        """GLiNER label strings for all entity types."""
        return [et.gliner_label for et in self.entity_types]

    def glirel_labels(self) -> list[str]:
        """GLiREL label strings for all relation types."""
        return [rt.glirel_label for rt in self.relation_types]

    @property
    def gliner_to_label(self) -> dict[str, str]:
        """Map GLiNER label → entity type label (for post-extraction normalization)."""
        return {et.gliner_label: et.label for et in self.entity_types}

    @property
    def glirel_to_type(self) -> dict[str, str]:
        """Map GLiREL label → relation type (for post-extraction normalization)."""
        return {rt.glirel_label: rt.type for rt in self.relation_types}

    def entity_type_names(self) -> set[str]:
        """Return the set of all entity type labels."""
        return {et.label for et in self.entity_types}

    def relation_type_names(self) -> set[str]:
        """Return the set of all relation type names."""
        return {rt.type for rt in self.relation_types}

    def entity_prompt_block(self) -> str:
        """Serialized entity type list for LLM/BAML prompt injection."""
        return "\n".join(f"- {et.label}: {et.description}" for et in self.entity_types)

    def relation_prompt_block(self) -> str:
        """Serialized relation type list for LLM/BAML prompt injection."""
        lines = []
        for rt in self.relation_types:
            heads = ", ".join(rt.head_labels)
            tails = ", ".join(rt.tail_labels)
            lines.append(f"- {rt.type} [{heads} → {tails}]: {rt.glirel_label}")
        return "\n".join(lines)

    def as_text(self) -> str:
        """Full schema string for text2cypher / BAML dynamic enum injection.

        Describes the generic :RELATES_TO {type} pattern with all valid type values
        so text2cypher can generate accurate Cypher for the generic-edge model.
        """
        lines = ["Graph schema:", "Nodes:"]
        for et in self.entity_types:
            lines.append(f"  (:Entity:{et.label} {{id, name, label, ...}})")
        lines.append("Relationships (all use :RELATES_TO with a type property):")
        for rt in self.relation_types:
            heads = "|".join(rt.head_labels)
            tails = "|".join(rt.tail_labels)
            lines.append(
                f"  ({heads})-[:RELATES_TO {{{{type: '{rt.type}', "
                f"description, score}}}}]->({tails})"
            )
        lines.append(f"\nValid type values: {sorted(self.relation_type_names())}")
        return "\n".join(lines)

    def allowed_pairs(self) -> set[tuple[str, str, str]]:
        """All ``(head_label, relation_type, tail_label)`` triples.

        permitted by this schema.
        """
        return {
            (h, rt.type, t)
            for rt in self.relation_types
            for h in rt.head_labels
            for t in rt.tail_labels
        }
