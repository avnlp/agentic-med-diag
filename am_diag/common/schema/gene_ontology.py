"""Authored Gene Ontology subgraph schema (BPO/CCO/MFN terms + is_a hierarchy)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


_GO_LABELS = {
    "GoBiologicalProcess": "Gene Ontology biological process term.",
    "GoCellularComponent": "Gene Ontology cellular component term.",
    "GoMolecularFunction": "Gene Ontology molecular function term.",
}

GENE_ONTOLOGY_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label=label,
            gliner_label=label,
            description=description,
            key_property="curie",
        )
        for label, description in _GO_LABELS.items()
    ],
    relation_types=[
        RelationType(
            type="PARENT",
            glirel_label="PARENT",
            head_labels=[label],
            tail_labels=[label],
            properties=["sources", "structural_label"],
        )
        for label in _GO_LABELS
    ],
)
