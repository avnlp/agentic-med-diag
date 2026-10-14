"""Fixtures for embedder tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure OPENAI_API_KEY is set so lazy client creation doesn't fail."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
