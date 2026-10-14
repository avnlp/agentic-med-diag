"""Authored OnSIDES subgraph schema (drug adverse drug reactions)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


ONSIDES_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Drug",
            gliner_label="Drug",
            description="Drug ingredient (RxNorm).",
            key_property="curie",
        ),
        EntityType(
            label="Phenotype",
            gliner_label="Phenotype",
            description="Adverse-effect phenotype (MedDRA).",
            key_property="curie",
        ),
    ],
    relation_types=[
        RelationType(
            type="ADVERSE_DRUG_REACTION",
            glirel_label="ADVERSE_DRUG_REACTION",
            head_labels=["Drug"],
            tail_labels=["Phenotype"],
            properties=["sources", "structural_label"],
        )
    ],
)
