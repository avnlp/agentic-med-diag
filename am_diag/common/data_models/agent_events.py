"""Typed streaming events emitted by the project-scoped agent.

``AgentEvent`` is a discriminated union (on the ``type`` field) so callers can
pattern-match the agent's progress while ``Project.stream`` yields events.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from am_diag.common.data_models.agent import (
    AgentPlan,
    ClinicalAnswer,
    SufficientContext,
)


class PlanEvent(BaseModel):
    """The planner produced a set of sub-questions."""

    type: Literal["plan"] = "plan"
    plan: AgentPlan


class ToolCallEvent(BaseModel):
    """A retrieval tool was invoked."""

    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: dict = Field(default_factory=dict)
    result_preview: str | None = None


class VerifierEvent(BaseModel):
    """The verifier assessed evidence sufficiency."""

    type: Literal["verifier"] = "verifier"
    assessment: SufficientContext


class TokenEvent(BaseModel):
    """A streamed chunk of model output text."""

    type: Literal["token"] = "token"
    text: str


class FinalEvent(BaseModel):
    """The final structured answer."""

    type: Literal["final"] = "final"
    answer: ClinicalAnswer


AgentEvent = Annotated[
    Union[PlanEvent, ToolCallEvent, VerifierEvent, TokenEvent, FinalEvent],
    Field(discriminator="type"),
]
