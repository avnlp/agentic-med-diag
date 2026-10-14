"""Canonical node-type, relation, and source vocabulary for graph loaders.

Ports the relevant members of OptimusKG's ``silver/nodes/constants.py``. This is
the authoritative source of the semantic relation names the raw builders emit
(per the design decision to author schemas from the datasets, not reverse-engineer
them from the released gold graph).
"""

from __future__ import annotations

from enum import StrEnum

from am_diag.common.schema.optimuskg import ABBREV_TO_LABEL


class Node(StrEnum):
    """Three-letter node-type abbreviations (OptimusKG convention)."""

    ANATOMY = "ANA"
    BIOLOGICAL_PROCESS = "BPO"
    CELLULAR_COMPONENT = "CCO"
    DISEASE = "DIS"
    DRUG = "DRG"
    EXPOSURE = "EXP"
    GENE = "GEN"
    MOLECULAR_FUNCTION = "MFN"
    PATHWAY = "PWY"
    PHENOTYPE = "PHE"


def structural_label(src: Node, dst: Node) -> str:
    """Return the structural edge label for an endpoint pair, e.g. ``"DRG-GEN"``."""
    return f"{src.value}-{dst.value}"


def label_for(node: Node) -> str:
    """Map a node abbreviation to its PascalCase entity label.

    Reuses ``ABBREV_TO_LABEL`` so structured nodes share label names with the
    released-gold loader without duplicating the mapping.
    """
    return ABBREV_TO_LABEL[node.value]


class Source(StrEnum):
    """Provenance database identifiers stamped onto structured edges."""

    BGEE = "BGEE"
    CTD = "CTD"
    DISGENET = "DISGENET"
    DRUG_BANK = "DRUG_BANK"
    DRUG_CENTRAL = "DRUG_CENTRAL"
    GO = "GO"
    HPO = "HPO"
    MONDO = "MONDO"
    ONSIDES = "ONSIDES"
    OPEN_TARGETS = "OPEN_TARGETS"
    PRIMEKG = "PRIMEKG"
    REACTOME = "REACTOME"
    UBERON = "UBERON"


class Relation(StrEnum):
    """Semantic relation names emitted by the raw builders.

    Accretes members as each loader's transform is ported; this is the
    authoritative relation vocabulary for the per-dataset schemas.
    """

    PARENT = "PARENT"
    # Bgee anatomy-gene expression.
    EXPRESSION_PRESENT = "EXPRESSION_PRESENT"
    EXPRESSION_ABSENT = "EXPRESSION_ABSENT"
    # DrugBank / Open Targets drug-gene.
    TARGET = "TARGET"
    ENZYME = "ENZYME"
    TRANSPORTER = "TRANSPORTER"
    CARRIER = "CARRIER"
    OTHER = "OTHER"
    # DrugBank / Open Targets drug-drug.
    DRUG_INTERACTION = "DRUG_INTERACTION"
    # DisGeNET / Open Targets associations.
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    # Open Targets disease-phenotype.
    HAS_PHENOTYPE = "HAS_PHENOTYPE"
    # DrugCentral / Open Targets drug-disease.
    INDICATION = "INDICATION"
    CONTRAINDICATION = "CONTRAINDICATION"
    # OnSIDES drug-phenotype.
    ADVERSE_DRUG_REACTION = "ADVERSE_DRUG_REACTION"
    # PrimeKG / PPI gene-gene.
    INTERACTS_WITH = "INTERACTS_WITH"
    # CTD environmental exposure.
    EXPOSURE_DISEASE = "EXPOSURE_DISEASE"
    EXPOSURE_GENE = "EXPOSURE_GENE"
    EXPOSURE_EXPOSURE = "EXPOSURE_EXPOSURE"
    EXPOSURE_BIOLOGICAL_PROCESS = "EXPOSURE_BIOLOGICAL_PROCESS"
    EXPOSURE_CELLULAR_COMPONENT = "EXPOSURE_CELLULAR_COMPONENT"
    EXPOSURE_MOLECULAR_FUNCTION = "EXPOSURE_MOLECULAR_FUNCTION"
