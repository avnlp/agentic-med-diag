"""Tests for the Workspace admin namespaces.

Graph creation builds the live graph eagerly, so the Neo4j driver is patched at
its construction boundary and connection env vars are injected. The vector store
and embedder are lazy and never connect during these tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from am_diag.api.errors import ConnectionError as WorkspaceConnectionError
from am_diag.api.errors import RegistryError
from am_diag.api.workspace import Workspace
from am_diag.common.data_models.connection import GraphConnection, VectorConnection


_DRIVER_TARGET = "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver"


@pytest.fixture
def mock_driver() -> MagicMock:
    """Return a MagicMock configured as a healthy Neo4j async driver."""
    driver = MagicMock()
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
    return driver


@pytest.fixture
def client(tmp_path) -> Workspace:
    """Return an Workspace backed by a temp registry and state dir."""
    return Workspace.open(
        registry_path=tmp_path / "registry.json", state_dir=tmp_path / "state"
    )


def _neo4j() -> GraphConnection:
    return GraphConnection(
        uri="bolt://h:7687",
        user_env="MED_NEO4J_USER",
        password_env="MED_NEO4J_PASSWORD",
    )


def _vector() -> VectorConnection:
    return VectorConnection(url_env="MED_QDRANT_URL", api_key_env="MED_QDRANT_KEY")


def _set_env(monkeypatch) -> None:
    monkeypatch.setenv("MED_NEO4J_USER", "neo4j")
    monkeypatch.setenv("MED_NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("MED_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MED_QDRANT_KEY", "")


class TestGraphAdmin:
    """Tests for client.graphs create/list/get and rollback behaviour."""

    async def test_create_round_trips(self, client, mock_driver, monkeypatch):
        """Verify a created graph appears in list and validates connectivity."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            rec = await client.graphs.create(
                "med", neo4j=_neo4j(), vector=_vector(), name="Medical"
            )
        assert rec.graph_id == "med"
        assert mock_driver.verify_connectivity.await_count == 1
        listed = await client.graphs.list()
        assert [g.graph_id for g in listed] == ["med"]
        await client.aclose()

    async def test_missing_env_var_rolls_back(self, client, mock_driver, monkeypatch):
        """Verify a missing secret env var raises and leaves no dangling record."""
        monkeypatch.delenv("MED_NEO4J_USER", raising=False)
        with (
            patch(_DRIVER_TARGET, return_value=mock_driver),
            pytest.raises(WorkspaceConnectionError),
        ):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        assert await client.graphs.list() == []

    async def test_duplicate_graph_id_raises(self, client, mock_driver, monkeypatch):
        """Verify creating the same graph id twice raises RegistryError."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
            with pytest.raises(RegistryError):
                await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        await client.aclose()


class TestProjectAdmin:
    """Tests for client.projects and project membership management."""

    async def test_create_project_and_add_graph(self, client, mock_driver, monkeypatch):
        """Verify a project can be created and a graph added to it."""
        _set_env(monkeypatch)
        with patch(_DRIVER_TARGET, return_value=mock_driver):
            await client.graphs.create("med", neo4j=_neo4j(), vector=_vector())
        await client.projects.create("clinical")
        await client.project("clinical").add_graph("med")
        assert await client.project("clinical").graphs() == ["med"]
        await client.aclose()

    async def test_create_project_unknown_graph_raises(self, client):
        """Verify referencing an unknown graph at project creation raises."""
        with pytest.raises(RegistryError):
            await client.projects.create("clinical", graph_ids=["ghost"])
