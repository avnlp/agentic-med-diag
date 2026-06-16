"""KG schema: entity/relation type definitions and the medical GraphRAG schema.

Usage::

    from am_diag.common.schema import GraphSchema, MEDICAL_GRAPHRAG_SCHEMA

    schema = MEDICAL_GRAPHRAG_SCHEMA
    print(schema.entity_prompt_block())
"""

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.common.schema.medical_graphrag import (
    ENTITY_TYPES,
    MEDICAL_GRAPHRAG_SCHEMA,
    RELATION_TYPES,
)


__all__ = [
    "EntityType",
    "RelationType",
    "GraphSchema",
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "MEDICAL_GRAPHRAG_SCHEMA",
]
