"""Translate DeepAgents/LangGraph stream events into the typed AgentEvent union.

The agent layer owns its event vocabulary; ``Project.stream`` just iterates the
mapped events. ``map_event`` is shape-based: tool and token events map by event
kind, and structured planner/verifier/final outputs map by their domain type, so
the mapping does not depend on internal node names.

The precise LangGraph event that carries each structured output should be
confirmed against a real ``astream_events("v2")`` trace. Unmapped events return
``None`` (skipped), so an unconfirmed shape never yields a wrong event.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from am_diag.common.data_models.agent import (
    AgentPlan,
    ClinicalAnswer,
    SufficientContext,
)
from am_diag.common.data_models.agent_events import (
    AgentEvent,
    FinalEvent,
    PlanEvent,
    TokenEvent,
    ToolCallEvent,
    VerifierEvent,
)


_STRUCTURED_EVENTS = frozenset({"on_chain_end", "on_chat_model_end"})


def map_event(event: dict[str, Any]) -> AgentEvent | None:
    """Map one LangGraph ``astream_events`` event to an :data:`AgentEvent`.

    Tool start/end, token streams, and structured planner/verifier/final outputs
    are mapped; unrecognized events return ``None`` so callers can skip them.
    """
    kind = event.get("event")
    data = event.get("data", {}) or {}
    if kind == "on_tool_start":
        tool_input = data.get("input") or {}
        return ToolCallEvent(
            tool=event.get("name", ""),
            args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
        )
    if kind == "on_tool_end":
        return ToolCallEvent(
            tool=event.get("name", ""),
            result_preview=str(data.get("output", ""))[:200],
        )
    if kind == "on_chat_model_stream":
        return _token_event(data.get("chunk"))
    if kind in _STRUCTURED_EVENTS:
        return _structured_event(data.get("output"))
    return None


def _token_event(chunk: Any) -> TokenEvent | None:
    """Build a TokenEvent from a model-stream chunk (object or dict)."""
    text = getattr(chunk, "content", None)
    if text is None and isinstance(chunk, dict):
        text = chunk.get("content")
    return TokenEvent(text=str(text)) if text else None


def _structured_event(output: Any) -> AgentEvent | None:
    """Map a structured node output to a plan/verifier/final event by type.

    Typed domain objects match directly; a ``structured_response`` dict is
    coerced to the final :class:`ClinicalAnswer` (the agent's response format).
    ``AgentPlan`` and ``SufficientContext`` match only as typed objects because
    both validate from an empty dict and a raw dict would be ambiguous.
    """
    if isinstance(output, ClinicalAnswer):
        return FinalEvent(answer=output)
    if isinstance(output, AgentPlan):
        return PlanEvent(plan=output)
    if isinstance(output, SufficientContext):
        return VerifierEvent(assessment=output)
    if isinstance(output, dict):
        return _final_from_dict(output)
    return None


def _final_from_dict(output: dict[str, Any]) -> FinalEvent | None:
    """Map a ``structured_response`` payload to the final ClinicalAnswer event."""
    structured = output.get("structured_response")
    if isinstance(structured, ClinicalAnswer):
        return FinalEvent(answer=structured)
    if isinstance(structured, dict):
        try:
            return FinalEvent(answer=ClinicalAnswer.model_validate(structured))
        except ValidationError:
            return None
    return None
