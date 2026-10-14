"""Authored CTD subgraph schema (environmental exposure relations)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


_PROPS = ["sources", "structural_label"]

_ENTITY_TYPES = [
    ("Exposure", "Environmental exposure / stressor (CTD, MeSH)."),
    ("Disease", "Disease or disorder (Mondo)."),
    ("Gene", "Human gene (Ensembl)."),
    ("GoBiologicalProcess", "Gene Ontology biological process term."),
    ("GoCellularComponent", "Gene Ontology cellular component term."),
    ("GoMolecularFunction", "Gene Ontology molecular function term."),
]

# (relation type, tail label) for each exposure edge family; head is always Exposure.
_RELATIONS = [
    ("EXPOSURE_DISEASE", "Disease"),
    ("EXPOSURE_GENE", "Gene"),
    ("EXPOSURE_EXPOSURE", "Exposure"),
    ("EXPOSURE_BIOLOGICAL_PROCESS", "GoBiologicalProcess"),
    ("EXPOSURE_CELLULAR_COMPONENT", "GoCellularComponent"),
    ("EXPOSURE_MOLECULAR_FUNCTION", "GoMolecularFunction"),
]

CTD_SCHEMA = GraphSchema(
    entity_types=[
        EntityType(
            label=label,
            gliner_label=label,
            description=description,
            key_property="curie",
        )
        for label, description in _ENTITY_TYPES
    ],
    relation_types=[
        RelationType(
            type=relation,
            glirel_label=relation,
            head_labels=["Exposure"],
            tail_labels=[tail],
            properties=_PROPS,
        )
        for relation, tail in _RELATIONS
    ],
)
