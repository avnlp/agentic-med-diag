"""Tests for QdrantSettings."""

from __future__ import annotations

import pytest

from am_diag.db.vector.settings import QdrantSettings


class TestQdrantSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        settings = QdrantSettings()
        assert settings.url == "http://localhost:6333"
        assert settings.api_key == ""

    def test_reads_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QDRANT_URL", "https://my-cluster.qdrant.io")
        monkeypatch.setenv("QDRANT_API_KEY", "secret")
        settings = QdrantSettings()
        assert settings.url == "https://my-cluster.qdrant.io"
        assert settings.api_key == "secret"

    def test_explicit_values_override_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QDRANT_URL", "http://ignored")
        settings = QdrantSettings(url="http://explicit:6333")
        assert settings.url == "http://explicit:6333"
