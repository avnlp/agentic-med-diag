"""KG schema: entity/relation type definitions and the medical GraphRAG schema.

Usage::

    from am_diag.common.schema import GraphSchema, MEDICAL_GRAPHRAG_SCHEMA

    schema = MEDICAL_GRAPHRAG_SCHEMA
    print(schema.entity_prompt_block())
"""

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.common.schema.bgee import BGEE_SCHEMA
from am_diag.common.schema.ctd import CTD_SCHEMA
from am_diag.common.schema.disgenet import DISGENET_SCHEMA
from am_diag.common.schema.drugbank import DRUGBANK_SCHEMA
from am_diag.common.schema.drugcentral import DRUGCENTRAL_SCHEMA
from am_diag.common.schema.gene_ontology import GENE_ONTOLOGY_SCHEMA
from am_diag.common.schema.hpo import HPO_SCHEMA
from am_diag.common.schema.medical_graphrag import (
    ENTITY_TYPES,
    MEDICAL_GRAPHRAG_SCHEMA,
    RELATION_TYPES,
)
from am_diag.common.schema.mondo import MONDO_SCHEMA
from am_diag.common.schema.onsides import ONSIDES_SCHEMA
from am_diag.common.schema.opentargets import OPEN_TARGETS_SCHEMA
from am_diag.common.schema.optimuskg import OPTIMUSKG_SCHEMA
from am_diag.common.schema.ppi import PPI_SCHEMA
from am_diag.common.schema.reactome import REACTOME_SCHEMA
from am_diag.common.schema.uberon import UBERON_SCHEMA


__all__ = [
    "EntityType",
    "RelationType",
    "GraphSchema",
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "MEDICAL_GRAPHRAG_SCHEMA",
    "OPTIMUSKG_SCHEMA",
    "UBERON_SCHEMA",
    "MONDO_SCHEMA",
    "HPO_SCHEMA",
    "GENE_ONTOLOGY_SCHEMA",
    "REACTOME_SCHEMA",
    "BGEE_SCHEMA",
    "DRUGBANK_SCHEMA",
    "DISGENET_SCHEMA",
    "PPI_SCHEMA",
    "ONSIDES_SCHEMA",
    "DRUGCENTRAL_SCHEMA",
    "CTD_SCHEMA",
    "OPEN_TARGETS_SCHEMA",
]
