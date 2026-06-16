"""Shared test fixtures for am_diag tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide minimal env vars so am_diag package-level imports don't fail."""
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("BAML_LLM_MODEL", "test-model")
    monkeypatch.setenv("BAML_LLM_ENDPOINT", "http://localhost:9999/v1")
    monkeypatch.setenv("BAML_LLM_API_KEY", "test-key")
    monkeypatch.setenv("GRAPH_DATABASE_URL", "bolt://localhost:7687")
    monkeypatch.setenv("GRAPH_DATABASE_USERNAME", "neo4j")
    monkeypatch.setenv("GRAPH_DATABASE_PASSWORD", "password")
    monkeypatch.setenv("DB_PATH", "/tmp/test_am_diag_db")
    monkeypatch.setenv("DB_NAME", "test_am_diag.db")
    monkeypatch.setenv("OPENAI_GENERIC_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_GENERIC_API_KEY", "test-key")
