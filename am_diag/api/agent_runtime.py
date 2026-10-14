"""Helpers bridging the DeepAgents runtime to the public agent surface.

Converts public inputs (str or ``Message`` list) into the agent's message
format, extracts a ``ClinicalAnswer`` from the agent output, and derives a stable
agent config hash for the answer cache. The LangGraph-to-``AgentEvent`` stream
mapping lives in ``am_diag.agents.events`` (the agent layer owns its events).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.message import Message


def to_messages(query_or_messages: str | list[Message]) -> list[dict[str, str]]:
    """Normalize a string or ``Message`` list into agent message dicts."""
    if isinstance(query_or_messages, str):
        return [{"role": "user", "content": query_or_messages}]
    return [{"role": m.role, "content": m.content} for m in query_or_messages]


def extract_clinical_answer(out: dict[str, Any]) -> ClinicalAnswer:
    """Extract a :class:`ClinicalAnswer` from raw agent output.

    Prefers the structured response; falls back to the last message content.

    Raises:
        RuntimeError: If neither a structured response nor a message is present.
    """
    structured = out.get("structured_response")
    if isinstance(structured, ClinicalAnswer):
        return structured
    if isinstance(structured, dict):
        return ClinicalAnswer.model_validate(structured)
    messages = out.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        if content is not None:
            return ClinicalAnswer(answer=str(content), answerable=True)
    raise RuntimeError(f"Unexpected agent output; keys: {list(out.keys())}")


def agent_config_hash(settings: Any, graphs: list[str]) -> str:
    """Return a stable hash of the agent config and target graph set."""
    payload = {
        "graphs": sorted(graphs),
        "settings": settings.model_dump() if hasattr(settings, "model_dump") else {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
