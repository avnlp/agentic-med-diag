"""Soft-defaults autouse fixture for integration tests.

Fills only **absent**, non-gating import-safety environment variables so that
package-level imports succeed without masking a service-skip gate.  Real values
from the process environment or repo-root ``.env`` always take precedence.
"""

from __future__ import annotations

import pytest

from tests._env import GATING_VARS, is_configured


# Import-safety placeholders ONLY. Real values from env/.env always win, and we
# never fill a GATING var (would mask the hook's service-skip). These keys are
# disjoint from GATING_VARS by construction; the GATING_VARS guard keeps that true
# as the map grows.
_IMPORT_SAFETY_DEFAULTS: dict[str, str] = {
    "LLM_MODEL": "test-model",
    "LLM_ENDPOINT": "http://localhost:9999/v1",
    "LLM_API_KEY": "test-key",
    "BAML_LLM_MODEL": "test-model",
    "BAML_LLM_ENDPOINT": "http://localhost:9999/v1",
    "BAML_LLM_API_KEY": "test-key",
    "DB_PATH": "/tmp/test_am_diag_db",
    "DB_NAME": "test_am_diag.db",
    "OPENAI_GENERIC_BASE_URL": "http://localhost:11434/v1",
    "OPENAI_GENERIC_API_KEY": "test-key",
}


@pytest.fixture(autouse=True)
def import_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fill only ABSENT, non-gating import-safety vars.

    This lets package imports succeed while real service config (env/.env)
    takes precedence.
    """
    for key, value in _IMPORT_SAFETY_DEFAULTS.items():
        if key in GATING_VARS:
            continue  # never mask a service-skip gate
        if not is_configured(key):
            monkeypatch.setenv(key, value)
