"""Agent data models for the DeepAgents clinical-QA harness.

All agent-facing types live here (Rule 1 from the overhaul plan):
no agent data model definitions outside ``common/data_models``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """A decomposed sub-question for parallel researcher fan-out."""

    text: str
    recipe: str = "hybrid_rrf"
    rationale: str = ""


class AgentPlan(BaseModel):
    """The planner's output: a list of sub-questions to research."""

    sub_questions: list[SubQuestion] = Field(default_factory=list)
    notes: str = ""


class EvidenceItem(BaseModel):
    """A single piece of evidence returned by a researcher subagent."""

    content: str
    source: str
    citation: str = ""
    score: float = 0.0


class SufficientContext(BaseModel):
    """Verifier's structured assessment of evidence sufficiency.

    Used by the ``sufficient_context`` subagent and the deterministic gate.
    """

    is_sufficient: bool = False
    coverage_score: int = 0
    evidence_depth: int = 0
    missing_pieces: list[str] = Field(default_factory=list)
    targeted_followups: list[SubQuestion] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class ClinicalAnswer(BaseModel):
    """The final structured answer from the orchestrator."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    caveats: list[str] = Field(default_factory=list)
    answerable: bool = True


__all__ = [
    "AgentPlan",
    "ClinicalAnswer",
    "EvidenceItem",
    "SubQuestion",
    "SufficientContext",
]
