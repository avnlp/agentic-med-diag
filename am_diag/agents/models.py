"""Model factory: ``build_chat_model(role)``."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from am_diag.agents.settings import AgentSettings


def build_chat_model(
    role: str = "orchestrator",
    settings: AgentSettings | None = None,
) -> ChatOpenAI:
    """Build a ``ChatOpenAI`` instance for a given role.

    If a per-role model name is set (e.g. ``researcher_model``), it overrides
    the default ``model``. All roles share the same ``base_url`` and
    ``api_key`` from settings.

    Args:
        role: The agent role: ``orchestrator``, ``planner``, ``researcher``,
            ``verifier``, or ``synthesis``.
        settings: ``AgentSettings`` instance. Defaults to ``AgentSettings()``.

    Returns:
        A configured ``ChatOpenAI`` instance.
    """
    s = settings or AgentSettings()
    model_name = getattr(s, f"{role}_model", None) or s.model
    return ChatOpenAI(
        model=model_name,
        base_url=s.base_url,
        api_key=s.api_key,  # type: ignore
        temperature=s.temperature,
        max_retries=5,
    )


__all__ = ["build_chat_model"]
