"""Tests for Neo4jSettings."""

from __future__ import annotations

from am_diag.db.graph.settings import Neo4jSettings


class TestNeo4jSettings:
    """Tests for Neo4jSettings."""

    def test_loads_uri_from_env(self, monkeypatch):
        """Verify URI is loaded from env variable."""
        monkeypatch.setenv("NEO4J_URI", "bolt://myhost:7687")
        settings = Neo4jSettings()
        assert settings.uri == "bolt://myhost:7687"

    def test_loads_user_and_password_from_env(self, monkeypatch):
        """Verify user and password are loaded from env."""
        monkeypatch.setenv("NEO4J_USER", "admin")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")
        settings = Neo4jSettings()
        assert settings.user == "admin"
        assert settings.password == "secret"

    def test_database_defaults_to_neo4j(self):
        """Verify database defaults to neo4j."""
        settings = Neo4jSettings()
        assert settings.database == "neo4j"

    def test_database_overridable_from_env(self, monkeypatch):
        """Verify database can be overridden via env."""
        monkeypatch.setenv("NEO4J_DATABASE", "mydb")
        settings = Neo4jSettings()
        assert settings.database == "mydb"

    def test_can_be_constructed_directly(self):
        """Verify settings can be constructed directly with kwargs."""
        settings = Neo4jSettings(uri="bolt://host:7687", user="u", password="p")
        assert settings.uri == "bolt://host:7687"
        assert settings.user == "u"
        assert settings.password == "p"

    def test_direct_construction_ignores_env(self, monkeypatch):
        """Verify direct construction overrides env variables."""
        monkeypatch.setenv("NEO4J_URI", "bolt://fromenv:7687")
        settings = Neo4jSettings(uri="bolt://direct:7687")
        assert settings.uri == "bolt://direct:7687"
