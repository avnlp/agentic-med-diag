"""Convenience entrypoint for answering clinical questions via the DeepAgents harness.

``answer_question(question)`` builds the agent, invokes it, and returns
a ``ClinicalAnswer``.
"""

from __future__ import annotations

from typing import Any

from am_diag.agents.agent import build_clinical_agent
from am_diag.agents.settings import AgentSettings
from am_diag.retrieval.search import SearchEngine


async def answer_question(
    question: str,
    *,
    search_engine: SearchEngine | None = None,
    settings: AgentSettings | None = None,
    stream: bool = False,
) -> Any:
    """Answer a clinical question using the DeepAgents harness.

    Args:
        question: The clinical question to answer.
        search_engine: Configured ``SearchEngine`` for retrieval.
        settings: ``AgentSettings``. Defaults to ``AgentSettings()``.
        stream: When ``True``, stream the response (not yet implemented).

    Returns:
        A ``ClinicalAnswer`` (or the raw graph output).

    Raises:
        RuntimeError: If the agent returns an unexpected structure.
    """
    s = settings or AgentSettings()
    agent = await build_clinical_agent(
        search_engine=search_engine,
        settings=s,
    )

    out = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": s.recursion_limit},
    )

    # Extract the structured response
    structured = out.get("structured_response")
    if structured is not None:
        return structured

    # Fallback: try to extract from messages
    messages = out.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "content"):
            from am_diag.common.data_models.agent import (  # noqa: PLC0415
                ClinicalAnswer,
            )

            return ClinicalAnswer(
                answer=last.content,
                answerable=True,
            )

    raise RuntimeError(f"Unexpected agent output structure. Keys: {list(out.keys())}")


__all__ = ["answer_question"]
