"""Authored DisGeNET subgraph schema (disease/phenotype-gene associations)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


_PROPERTIES = ["sources", "structural_label", "evidence_index"]

DISGENET_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Disease",
            gliner_label="Disease",
            description="Disease or disorder (Open Targets canonical).",
            key_property="curie",
        ),
        EntityType(
            label="Phenotype",
            gliner_label="Phenotype",
            description="Phenotypic feature (Open Targets canonical).",
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
            type="ASSOCIATED_WITH",
            glirel_label="ASSOCIATED_WITH",
            head_labels=["Disease", "Phenotype"],
            tail_labels=["Gene"],
            properties=_PROPERTIES,
        )
    ],
)
