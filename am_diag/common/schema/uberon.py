"""Authored Uberon subgraph schema (anatomy identity + is_a hierarchy)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


UBERON_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="AnatomicalStructure",
            gliner_label="AnatomicalStructure",
            description="Anatomical structure or tissue (Uberon).",
            key_property="curie",
        )
    ],
    relation_types=[
        RelationType(
            type="PARENT",
            glirel_label="PARENT",
            head_labels=["AnatomicalStructure"],
            tail_labels=["AnatomicalStructure"],
            properties=["sources", "structural_label"],
        )
    ],
)
