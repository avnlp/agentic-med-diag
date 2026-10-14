"""Authored PrimeKG/PPI subgraph schema (gene-gene protein interactions)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


PPI_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label="Gene",
            gliner_label="Gene",
            description="Human gene (Ensembl).",
            key_property="curie",
        )
    ],
    relation_types=[
        RelationType(
            type="INTERACTS_WITH",
            glirel_label="INTERACTS_WITH",
            head_labels=["Gene"],
            tail_labels=["Gene"],
            properties=["sources", "structural_label", "databases"],
        )
    ],
)
