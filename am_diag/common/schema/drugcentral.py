"""Authored DrugCentral subgraph schema (drug-disease and drug-phenotype)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


_RELATIONS = ["INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE"]

DRUGCENTRAL_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Drug",
            gliner_label="Drug",
            description="Drug or chemical compound (ChEMBL).",
            key_property="curie",
        ),
        EntityType(
            label="Disease",
            gliner_label="Disease",
            description="Disease or disorder (Open Targets canonical).",
            key_property="curie",
        ),
        EntityType(
            label="Phenotype",
            gliner_label="Phenotype",
            description="Phenotypic feature (HPO).",
            key_property="curie",
        ),
    ],
    relation_types=[
        RelationType(
            type=relation,
            glirel_label=relation,
            head_labels=["Drug"],
            tail_labels=["Disease", "Phenotype"],
            properties=[
                "sources",
                "structural_label",
                "asserted_relations",
                "relation_conflict",
            ],
        )
        for relation in _RELATIONS
    ],
)
