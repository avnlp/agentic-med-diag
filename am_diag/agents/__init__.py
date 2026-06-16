"""DeepAgents-based clinical QA harness.

The agent loop follows the Google-faithful multi-role topology:
Orchestrator → Planner → parallel Researchers → Verifier → Synthesis.
BAML is kept only inside retrieval/ingestion tools.

Usage::

    from am_diag.agents import build_clinical_agent, answer_question

    agent = await build_clinical_agent(search_engine=engine)
    answer = await agent.ainvoke(...)
"""

from __future__ import annotations

from am_diag.agents.agent import build_clinical_agent
from am_diag.agents.run import answer_question
from am_diag.agents.settings import AgentSettings


__all__ = [
    "AgentSettings",
    "answer_question",
    "build_clinical_agent",
]
