"""Authored Bgee subgraph schema (anatomy-gene expression)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


_EXPRESSION_PROPERTIES = [
    "sources",
    "structural_label",
    "expression_rank",
    "call_quality",
]

BGEE_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="AnatomicalStructure",
            gliner_label="AnatomicalStructure",
            description="Anatomical structure or tissue (Uberon).",
            key_property="curie",
        ),
        EntityType(
            label="Gene",
            gliner_label="Gene",
            description="Human gene (Ensembl).",
            key_property="curie",
        ),
    ],
    relation_types=[
        RelationType(
            type="EXPRESSION_PRESENT",
            glirel_label="EXPRESSION_PRESENT",
            head_labels=["AnatomicalStructure"],
            tail_labels=["Gene"],
            properties=_EXPRESSION_PROPERTIES,
        ),
        RelationType(
            type="EXPRESSION_ABSENT",
            glirel_label="EXPRESSION_ABSENT",
            head_labels=["AnatomicalStructure"],
            tail_labels=["Gene"],
            properties=_EXPRESSION_PROPERTIES,
        ),
    ],
)
