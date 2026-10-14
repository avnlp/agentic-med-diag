"""Authored Reactome subgraph schema (pathway identity + hierarchy)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


REACTOME_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Pathway",
            gliner_label="Pathway",
            description="Biological pathway (Reactome).",
            key_property="curie",
        )
    ],
    relation_types=[
        RelationType(
            type="PARENT",
            glirel_label="PARENT",
            head_labels=["Pathway"],
            tail_labels=["Pathway"],
            properties=["sources", "structural_label"],
        )
    ],
)
