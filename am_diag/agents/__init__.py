"""DeepAgents-based clinical QA harness.

The agent loop follows the Google-faithful multi-role topology:
Orchestrator → Planner → parallel Researchers → Verifier → Synthesis.
BAML is kept only inside retrieval/ingestion tools.

Usage::

    from am_diag.agents import build_clinical_agent

    agent = await build_clinical_agent(search_engine=engine)
    answer = await agent.ainvoke(...)

The public answer/stream/answer_batch surface lives on ``am_diag.Workspace``'s
project handles (``client.project(...).answer(...)``).
"""

from __future__ import annotations

from am_diag.agents.agent import build_clinical_agent
from am_diag.agents.settings import AgentSettings


__all__ = [
    "AgentSettings",
    "build_clinical_agent",
]
