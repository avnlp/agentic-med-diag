"""OptimusKG biomedical knowledge-graph schema (union of all source subgraphs).

OptimusKG nodes use 10 ``input_label`` abbreviations and edges 27 structural
``HEAD-TAIL`` labels. This module maps those onto am_diag PascalCase entity labels
and generates the ``GraphSchema`` used by the released-OptimusKG loader (and, as a
subset, by future per-source subgraph loaders).

The per-edge **semantic** relation (``Relation.type``, e.g. ``EXPRESSED_IN``,
``TARGET``) lives in the gold ``relation`` column, which the loader maps to
``type``, so the exact set of relation types is data-derived.
``SEMANTIC_RELATIONS`` is populated once from the gold ``edges`` table and
committed here; until then each structural label contributes a single placeholder
type (the structural label with ``-`` replaced by ``_``).
Relation types are hints for text2cypher and ``allowed_pairs`` only; they do not
affect ingestion (``setup_schema`` keys constraints on entity labels), so the
placeholder is safe.
"""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


# OptimusKG node ``input_label`` abbreviation -> am_diag PascalCase entity label.
ABBREV_TO_LABEL: dict[str, str] = {
    "ANA": "AnatomicalStructure",
    "BPO": "GoBiologicalProcess",
    "CCO": "GoCellularComponent",
    "DIS": "Disease",
    "DRG": "Drug",
    "EXP": "Exposure",
    "GEN": "Gene",
    "MFN": "GoMolecularFunction",
    "PHE": "Phenotype",
    "PWY": "Pathway",
}

_LABEL_DESCRIPTIONS: dict[str, str] = {
    "AnatomicalStructure": "Anatomical structure or tissue (UBERON).",
    "GoBiologicalProcess": "Gene Ontology biological process term.",
    "GoCellularComponent": "Gene Ontology cellular component term.",
    "Disease": "Disease or disorder (MONDO / UMLS).",
    "Drug": "Drug or chemical compound (ChEMBL / DrugBank).",
    "Exposure": "Environmental exposure (CTD).",
    "Gene": "Human gene or genetic locus (Ensembl / HGNC).",
    "GoMolecularFunction": "Gene Ontology molecular function term.",
    "Phenotype": "Phenotypic feature (HPO).",
    "Pathway": "Biological pathway (Reactome).",
}

# 27 OptimusKG edge ``input_label`` values (structural ``HEAD-TAIL``).
STRUCTURAL_LABELS: list[str] = [
    "ANA-GEN",
    "ANA-ANA",
    "EXP-DIS",
    "EXP-EXP",
    "EXP-GEN",
    "DIS-GEN",
    "DIS-DIS",
    "PHE-GEN",
    "DRG-GEN",
    "DRG-DRG",
    "DRG-DIS",
    "DRG-BPO",
    "DRG-PHE",
    "BPO-GEN",
    "CCO-GEN",
    "MFN-GEN",
    "PWY-PWY",
    "PWY-GEN",
    "EXP-BPO",
    "EXP-MFN",
    "EXP-CCO",
    "CCO-CCO",
    "BPO-BPO",
    "MFN-MFN",
    "DIS-PHE",
    "PHE-PHE",
    "GEN-GEN",
]

# Distinct semantic ``relation`` values per structural label, enumerated from the
# gold ``edges`` table and pasted here. Empty means "use the structural-label
# placeholder" (see module docstring).
SEMANTIC_RELATIONS: dict[str, list[str]] = {}

# Properties every structured relation may carry (queryable on the edge).
_RELATION_PROPERTIES = ["undirected", "sources", "structural_label"]


def _endpoints(structural_label: str) -> tuple[str, str]:
    """Return the ``(head_label, tail_label)`` for a structural ``HEAD-TAIL`` label."""
    head, tail = structural_label.split("-")
    return ABBREV_TO_LABEL[head], ABBREV_TO_LABEL[tail]


def _build_entity_types() -> list[EntityType]:
    """Build one EntityType per unique am_diag label (identity keyed on curie)."""
    labels = list(dict.fromkeys(ABBREV_TO_LABEL.values()))
    return [
        EntityType(
            label=label,
            gliner_label=label,
            description=_LABEL_DESCRIPTIONS[label],
            key_property="curie",
        )
        for label in labels
    ]


def _build_relation_types() -> list[RelationType]:
    """Build RelationTypes from structural labels and their semantic relations."""
    relation_types: list[RelationType] = []
    for structural in STRUCTURAL_LABELS:
        head_label, tail_label = _endpoints(structural)
        semantics = SEMANTIC_RELATIONS.get(structural) or [structural.replace("-", "_")]
        for semantic in semantics:
            relation_types.append(
                RelationType(
                    type=semantic,
                    glirel_label=semantic,
                    head_labels=[head_label],
                    tail_labels=[tail_label],
                    properties=_RELATION_PROPERTIES,
                )
            )
    return relation_types


ENTITY_TYPES: list[EntityType] = _build_entity_types()
RELATION_TYPES: list[RelationType] = _build_relation_types()

OPTIMUSKG_SCHEMA = GraphSchema(
    entity_types=ENTITY_TYPES,
    relation_types=RELATION_TYPES,
)

__all__ = [
    "ABBREV_TO_LABEL",
    "STRUCTURAL_LABELS",
    "SEMANTIC_RELATIONS",
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "OPTIMUSKG_SCHEMA",
]
