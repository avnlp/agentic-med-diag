"""Subagent configurations for the clinical-QA DeepAgents harness.

Each config follows the ``deepagents`` ``SubAgent`` TypedDict:
- name (required): Unique identifier.
- description (required): Action-oriented description for the orchestrator.
- system_prompt (required): Instructions for the subagent.
- tools (optional): Subagent-specific tools.
- model (optional): Per-subagent model override.
- response_format (optional): Structured output schema.
"""

from __future__ import annotations

from typing import Any

from am_diag.agents.models import build_chat_model
from am_diag.agents.prompts import (
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    VERIFIER_PROMPT,
)
from am_diag.agents.settings import AgentSettings
from am_diag.agents.tools import make_retrieval_tools
from am_diag.retrieval.search import SearchEngine


def build_researcher_subagent(
    search_engine: SearchEngine,
    settings: AgentSettings | None = None,
) -> dict[str, Any]:
    """Build the Researcher subagent configuration.

    The Researcher has retrieval tools and is spawned in parallel for each
    sub-question. It writes raw hits to the evidence directory and returns
    distilled evidence items.
    """
    s = settings or AgentSettings()
    tools = make_retrieval_tools(search_engine)
    model = build_chat_model("researcher", s)

    return {
        "name": "researcher",
        "description": "Retrieve and distill grounded evidence for ONE sub-question. "
        "Use retrieval tools to search the knowledge graph, "
        "then return structured evidence items with citations.",
        "system_prompt": RESEARCHER_PROMPT,
        "tools": tools,
        "model": model if s.researcher_model else None,
    }


def build_planner_subagent(
    settings: AgentSettings | None = None,
) -> dict[str, Any]:
    """Build the Planner subagent configuration.

    The Planner decomposes the clinical question into focused sub-questions,
    each mapped to a retrieval recipe. No retrieval tools — it operates
    purely on the question text.
    """
    from am_diag.common.data_models.agent import AgentPlan  # noqa: PLC0415

    s = settings or AgentSettings()
    model = build_chat_model("planner", s)

    return {
        "name": "planner",
        "description": "Decompose the clinical question into targeted sub-questions, "
        "each mapped to a retrieval recipe.",
        "system_prompt": PLANNER_PROMPT,
        "model": model if s.planner_model else None,
        "response_format": AgentPlan,
    }


def build_verifier_subagent(
    settings: AgentSettings | None = None,
) -> dict[str, Any]:
    """Build the Verifier / Sufficient-Context subagent configuration.

    The Verifier has no tools — it reads the gathered evidence and returns
    a structured ``SufficientContext`` assessment.
    """
    from am_diag.common.data_models.agent import (  # noqa: PLC0415
        SufficientContext,
    )

    s = settings or AgentSettings()
    model = build_chat_model("verifier", s)

    return {
        "name": "sufficient_context",
        "description": "Judge whether gathered evidence sufficiently answers the "
        "question; return actionable gaps if not.",
        "system_prompt": VERIFIER_PROMPT,
        "model": model if s.verifier_model else None,
        "response_format": SufficientContext,
    }


__all__ = [
    "build_planner_subagent",
    "build_researcher_subagent",
    "build_verifier_subagent",
]
