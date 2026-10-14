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
        if _is_nested(event.get("metadata")):
            return None
        return _token_event(data.get("chunk"))
    if kind in _STRUCTURED_EVENTS:
        return _structured_event(data.get("output"))
    return None


def _is_nested(metadata: dict[str, Any] | None) -> bool:
    """Return whether an event comes from a subagent graph.

    Subagents run inside a tool node of the orchestrator graph, so their
    checkpoint namespace has more than one ``|``-separated segment. Only the
    orchestrator's own output is user-facing.
    """
    metadata = metadata or {}
    namespace = metadata.get("langgraph_checkpoint_ns") or metadata.get(
        "checkpoint_ns", ""
    )
    return "|" in namespace


def _text_from_content(content: Any) -> str:
    """Return the text of message content, which may be a list of blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _token_event(chunk: Any) -> TokenEvent | None:
    """Build a TokenEvent from a model-stream chunk (object or dict)."""
    content = getattr(chunk, "content", None)
    if content is None and isinstance(chunk, dict):
        content = chunk.get("content")
    text = _text_from_content(content)
    return TokenEvent(text=text) if text else None


def _typed_event(value: Any) -> AgentEvent | None:
    """Map a typed domain object to its plan/verifier/final event."""
    if isinstance(value, ClinicalAnswer):
        return FinalEvent(answer=value)
    if isinstance(value, AgentPlan):
        return PlanEvent(plan=value)
    if isinstance(value, SufficientContext):
        return VerifierEvent(assessment=value)
    return None


def _structured_event(output: Any) -> AgentEvent | None:
    """Map a structured node output to a plan/verifier/final event by type.

    Typed domain objects match directly, either as the output itself or as the
    ``structured_response`` of a graph state. A ``structured_response`` dict is
    coerced to the final :class:`ClinicalAnswer` (the orchestrator's response
    format). ``AgentPlan`` and ``SufficientContext`` match only as typed objects
    because both validate from an empty dict and a raw dict would be ambiguous.
    """
    if not isinstance(output, dict):
        return _typed_event(output)
    structured = output.get("structured_response")
    if isinstance(structured, dict):
        try:
            return FinalEvent(answer=ClinicalAnswer.model_validate(structured))
        except ValidationError:
            return None
    return _typed_event(structured)
