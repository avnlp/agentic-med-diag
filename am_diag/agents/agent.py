"""Clinical QA agent factory using the DeepAgents SDK.

``build_clinical_agent()`` is the single entry point that wires together the
Orchestrator, Planner, Researcher, and Verifier subagents into a deep agent.
All deepagents API specifics are contained in this file so an API version
bump touches at most this file and ``subagents.py``.
"""

from __future__ import annotations

from typing import Any

from am_diag.agents.models import build_chat_model
from am_diag.agents.prompts import ORCHESTRATOR_PROMPT
from am_diag.agents.settings import AgentSettings
from am_diag.agents.subagents import (
    build_planner_subagent,
    build_researcher_subagent,
    build_verifier_subagent,
)
from am_diag.retrieval.search import SearchEngine


async def build_clinical_agent(
    search_engine: SearchEngine | None = None,
    settings: AgentSettings | None = None,
) -> Any:
    """Build and return a compiled DeepAgents clinical QA agent.

    Args:
        search_engine: Configured ``SearchEngine`` for retrieval.
            If ``None``, retrieval tools will not be available.
        settings: ``AgentSettings``. Defaults to ``AgentSettings()``.

    Returns:
        A compiled ``CompiledStateGraph`` (deepagents) ready for invocation.
    """
    from deepagents import create_deep_agent  # noqa: PLC0415

    from am_diag.common.data_models.agent import ClinicalAnswer  # noqa: PLC0415

    s = settings or AgentSettings()
    se = search_engine

    # Build orchestrator model
    orchestrator_model = build_chat_model("orchestrator", s)

    # Build subagents
    subagent_configs = [
        build_planner_subagent(s),
        build_verifier_subagent(s),
    ]

    if se is not None:
        subagent_configs.append(build_researcher_subagent(se, s))

    # Construct the deep agent
    return create_deep_agent(
        model=orchestrator_model,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=subagent_configs,  # type: ignore
        response_format=ClinicalAnswer,
    )


__all__ = ["build_clinical_agent"]
