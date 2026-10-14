"""Authored Mondo subgraph schema (disease identity + is_a hierarchy)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


MONDO_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Disease",
            gliner_label="Disease",
            description="Disease or disorder (Mondo).",
            key_property="curie",
        )
    ],
    relation_types=[
        RelationType(
            type="PARENT",
            glirel_label="PARENT",
            head_labels=["Disease"],
            tail_labels=["Disease"],
            properties=["sources", "structural_label"],
        )
    ],
)
