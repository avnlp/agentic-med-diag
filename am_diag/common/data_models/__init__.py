"""am_diag graph-storable data types and shared models.

All domain data structures live here — no model definitions outside this package
(except BAML-generated code and pydantic_settings subclasses).

Usage::
    from am_diag.common.data_models import Entity, Relation, GraphSchema, SearchResult
"""

from __future__ import annotations

from am_diag.common.data_models.agent import (
    AgentPlan,
    ClinicalAnswer,
    EvidenceItem,
    SubQuestion,
    SufficientContext,
)
from am_diag.common.data_models.annotations import Dedup, Embeddable, LLMContext
from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.cluster import ItemCluster, ResolvedAliasMap
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport, Finding
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import Distance, ExtractorName
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.mcq import MCQSample
from am_diag.common.data_models.metadata import MetaData
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.qa import OpenEndedSample
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.rubric import (
    ConsensusCriterion,
    ConversationTurn,
    RARMedSample,
    RubricCriterion,
    RubricSample,
)
from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.common.data_models.search import SearchResult
from am_diag.common.data_models.vector_record import VectorHit, VectorRecord


QASample = MCQSample | RubricSample | RARMedSample | OpenEndedSample

__all__ = [
    "AgentPlan",
    "Chunk",
    "ClinicalAnswer",
    "Community",
    "CommunityReport",
    "ConsensusCriterion",
    "ConversationTurn",
    "DataPoint",
    "Dedup",
    "Distance",
    "Document",
    "Embeddable",
    "Entity",
    "EntityType",
    "EvidenceItem",
    "ExtractorName",
    "Finding",
    "GraphSchema",
    "ItemCluster",
    "KnowledgeGraph",
    "LLMContext",
    "MCQSample",
    "MetaData",
    "OpenEndedSample",
    "Provenance",
    "QASample",
    "RARMedSample",
    "Relation",
    "RelationType",
    "ResolvedAliasMap",
    "RubricCriterion",
    "RubricSample",
    "SearchResult",
    "SubQuestion",
    "SufficientContext",
    "VectorHit",
    "VectorRecord",
]
