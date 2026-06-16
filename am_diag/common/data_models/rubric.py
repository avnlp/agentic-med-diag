"""Rubric-scored evaluation sample models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    """A single message in a multi-turn HealthBench conversation.

    Captures the `role` (e.g. `"user"` / `"assistant"`) and
    `content` of each exchange. Used inside RubricSample to
    preserve the dialog structure for rubric scoring.
    """

    role: str
    content: str


class RubricCriterion(BaseModel):
    """A single rubric criterion from a HealthBench scoring rubric."""

    criterion: str
    points: int
    tags: list[str]
    axis: str | None
    level: str | None
    cluster: str | None


class ConsensusCriterion(BaseModel):
    """A consensus-driven rubric criterion for HealthBench scoring.

    HealthBench clusters rubric criteria by theme and behavior category.
    When a rubric item has a known `cluster` tag, the corresponding
    consensus criterion is looked up from `hb_consensus_criteria.json`
    and stored on the sample for automated scoring pipelines.
    """

    theme: str
    behavior_category: str
    criterion: str


class RubricSample(BaseModel):
    """A HealthBench multi-turn rubric-scored sample."""

    sample_id: str
    conversation: list[ConversationTurn]
    rubrics: list[RubricCriterion]
    theme: str | None
    variant: str
    dataset: str = "healthbench"
    criteria: list[str]
    points_list: list[int]
    axes: list[str]
    criterion_ids: list[str]
    consensus_criteria: list[ConsensusCriterion | None]
    question: str = ""
    question_stem: str = ""
    answer: str = ""
    metadata: dict[str, Any]


class RARMedSample(BaseModel):
    """A RAR-Medicine open-ended rubric-scored sample."""

    sample_id: str
    question: str
    question_stem: str = ""
    reference_answer: str
    rubrics: list[dict[str, Any]]
    dataset: str = "rar_med"
    answer: str
    metadata: dict[str, Any]
