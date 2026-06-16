"""Fixtures for graph_db tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def neo4j_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide NEO4J_* env vars so Neo4jSettings can be instantiated without .env."""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
