"""Shared enumerations for the am_diag data layer."""

from enum import Enum


class ExtractorName(str, Enum):
    """Identifies which extractor produced a given raw entity or relation."""

    GLINER = "gliner"
    GLIREL = "glirel"
    LLM = "llm"
    FUSION = "fusion"
    COMMUNITY_DETECTION = "community_detection"
    COMMUNITY_REPORT = "community_report"


class Distance(str, Enum):
    """Vector distance metric. Values match Qdrant/Weaviate conventions."""

    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"
