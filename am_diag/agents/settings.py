"""Settings for the DeepAgents clinical-QA harness — env-overridable with AGENT_ prefix.

Includes endpoint configuration (base_url, api_key, model), per-role model
overrides, loop control, and sufficiency thresholds.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Configuration for the DeepAgents clinical-QA harness.

    All fields are overridable via environment variables with the ``AGENT_``
    prefix (e.g. ``AGENT_BASE_URL=https://api.openai.com/v1``).
    """

    model_config = {"env_prefix": "AGENT_", "env_file": ".env", "extra": "ignore"}

    # Endpoint — shared across all roles
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"

    # Per-role model overrides (optional — default to ``model``)
    orchestrator_model: str | None = None
    planner_model: str | None = None
    researcher_model: str | None = None
    verifier_model: str | None = None
    synthesis_model: str | None = None

    # Temperature
    temperature: float = 0.0

    # Loop control
    max_iterations: int = 5
    recursion_limit: int = 99

    # Parallelism
    max_parallel_researchers: int = 5

    # Sufficiency thresholds
    coverage_threshold: int = 8
    evidence_threshold: int = 6

    # Evidence directory (virtual FS path for DeepAgents)
    request_evidence_dir: str = "/evidence"


__all__ = ["AgentSettings"]
