"""Authored Open Targets subgraph schema (the multi-relation target-disease graph).

Open Targets is the richest single source: target-disease/phenotype/anatomy/GO
associations, the disease hierarchy and disease-phenotype links, drug indications,
and drug-target mechanisms (the action-type vocabulary).
"""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


# Open Targets drug mechanism action types (normalized to UPPER_SNAKE) -> DRG-GEN.
DRUG_GENE_ACTIONS = [
    "ACTIVATOR",
    "AGONIST",
    "ALLOSTERIC_ANTAGONIST",
    "ANTAGONIST",
    "BINDING_AGENT",
    "BLOCKER",
    "DEGRADER",
    "INHIBITOR",
    "INVERSE_AGONIST",
    "MODULATOR",
    "NEGATIVE_ALLOSTERIC_MODULATOR",
    "NEGATIVE_MODULATOR",
    "OPENER",
    "OTHER",
    "PARTIAL_AGONIST",
    "POSITIVE_ALLOSTERIC_MODULATOR",
    "POSITIVE_MODULATOR",
    "RELEASING_AGENT",
    "STABILISER",
    "SUBSTRATE",
]

_GENE_ASSOCIATION_HEADS = [
    "Disease",
    "Phenotype",
    "AnatomicalStructure",
    "GoBiologicalProcess",
    "GoCellularComponent",
    "GoMolecularFunction",
]

_ENTITY_TYPES = [
    ("Gene", "Human gene (Ensembl)."),
    ("Disease", "Disease or disorder (Open Targets / Mondo / EFO)."),
    ("Phenotype", "Phenotypic feature (HPO)."),
    ("Drug", "Drug or chemical compound (ChEMBL)."),
    ("AnatomicalStructure", "Anatomical structure or tissue (Uberon)."),
    ("GoBiologicalProcess", "Gene Ontology biological process term."),
    ("GoCellularComponent", "Gene Ontology cellular component term."),
    ("GoMolecularFunction", "Gene Ontology molecular function term."),
]


def _relation(rtype: str, heads: list[str], tails: list[str]) -> RelationType:
    return RelationType(
        type=rtype,
        glirel_label=rtype,
        head_labels=heads,
        tail_labels=tails,
        properties=["sources", "structural_label"],
    )


OPEN_TARGETS_SCHEMA = GraphSchema(
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
        _relation("ASSOCIATED_WITH", _GENE_ASSOCIATION_HEADS, ["Gene"]),
        _relation("PARENT", ["Disease"], ["Disease"]),
        _relation("HAS_PHENOTYPE", ["Disease"], ["Phenotype"]),
        _relation("INDICATION", ["Drug"], ["Disease"]),
        *[_relation(action, ["Drug"], ["Gene"]) for action in DRUG_GENE_ACTIONS],
    ],
)
