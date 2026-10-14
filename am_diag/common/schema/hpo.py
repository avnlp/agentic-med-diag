"""Authored HPO subgraph schema (phenotype identity + is_a hierarchy)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


HPO_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Phenotype",
            gliner_label="Phenotype",
            description="Phenotypic feature (Human Phenotype Ontology).",
            key_property="curie",
        )
    ],
    relation_types=[
        RelationType(
            type="PARENT",
            glirel_label="PARENT",
            head_labels=["Phenotype"],
            tail_labels=["Phenotype"],
            properties=["sources", "structural_label"],
        )
    ],
)
