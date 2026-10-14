"""Authored DrugBank subgraph schema (drug-target and drug-drug interactions)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


_DRG_GEN = ["sources", "structural_label"]
_DRG_DRG = ["sources", "structural_label", "interaction_description"]

_DRUG_GENE_RELATIONS = ["TARGET", "ENZYME", "TRANSPORTER", "CARRIER", "OTHER"]

DRUGBANK_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Drug",
            gliner_label="Drug",
            description="Drug or chemical compound (DrugBank / ChEMBL).",
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
            type=relation,
            glirel_label=relation,
            head_labels=["Drug"],
            tail_labels=["Gene"],
            properties=_DRG_GEN,
        )
        for relation in _DRUG_GENE_RELATIONS
    ]
    + [
        RelationType(
            type="SYNERGISTIC_INTERACTION",
            glirel_label="SYNERGISTIC_INTERACTION",
            head_labels=["Drug"],
            tail_labels=["Drug"],
            properties=_DRG_DRG,
        )
    ],
)
