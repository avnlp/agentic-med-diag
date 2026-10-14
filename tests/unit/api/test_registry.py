"""Tests for the file-backed Registry.

Exercises CRUD, duplicate/not-found errors, atomic persistence across reopen,
and the invariant that only env-var names (never secrets) are written to disk.
"""

from __future__ import annotations

import json

import pytest

from am_diag.api.errors import (
    GraphNotFoundError,
    ProjectNotFoundError,
    RegistryError,
)
from am_diag.api.registry import Registry
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.project import ProjectRecord


def _graph(graph_id: str = "med") -> GraphRecord:
    """Build a minimal GraphRecord for tests."""
    return GraphRecord(
        graph_id=graph_id,
        neo4j=GraphConnection(
            uri="bolt://h:7687",
            user_env="MED_NEO4J_USER",
            password_env="MED_NEO4J_PASSWORD",
        ),
        vector=VectorConnection(url_env="MED_QDRANT_URL", api_key_env="MED_QDRANT_KEY"),
    )


class TestRegistry:
    """Tests for Registry persistence and CRUD."""

    def test_add_and_get_graph(self, tmp_path):
        """Verify a stored graph can be read back."""
        reg = Registry(tmp_path / "registry.json")
        reg.add_graph(_graph())
        assert reg.get_graph("med").graph_id == "med"

    def test_duplicate_graph_raises(self, tmp_path):
        """Verify adding a duplicate graph id raises RegistryError."""
        reg = Registry(tmp_path / "registry.json")
        reg.add_graph(_graph())
        with pytest.raises(RegistryError):
            reg.add_graph(_graph())

    def test_get_missing_graph_raises(self, tmp_path):
        """Verify reading an unknown graph raises GraphNotFoundError."""
        reg = Registry(tmp_path / "registry.json")
        with pytest.raises(GraphNotFoundError):
            reg.get_graph("nope")

    def test_persistence_across_reopen(self, tmp_path):
        """Verify records survive a fresh Registry opened on the same file."""
        path = tmp_path / "registry.json"
        reg = Registry(path)
        reg.add_graph(_graph())
        reg.add_project(ProjectRecord(project_id="clinical", graph_ids=["med"]))

        reopened = Registry(path)
        assert reopened.get_graph("med").graph_id == "med"
        assert reopened.get_project("clinical").graph_ids == ["med"]

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path):
        """Verify the atomic temp file is replaced, not left behind."""
        path = tmp_path / "registry.json"
        reg = Registry(path)
        reg.add_graph(_graph())
        assert path.exists()
        assert not (path.with_suffix(path.suffix + ".tmp")).exists()

    def test_failed_replace_preserves_previous_registry(self, tmp_path, monkeypatch):
        """Verify a failed write leaves the file and the in-memory state unchanged."""
        path = tmp_path / "registry.json"
        reg = Registry(path)
        reg.add_graph(_graph("med"))
        before = path.read_text("utf-8")

        def fail_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr("am_diag.api.registry.os.replace", fail_replace)
        with pytest.raises(OSError, match="disk full"):
            reg.add_graph(_graph("drug"))

        assert path.read_text("utf-8") == before
        assert [g.graph_id for g in reg.list_graphs()] == ["med"]
        with pytest.raises(GraphNotFoundError):
            reg.get_graph("drug")

    def test_stale_instance_does_not_overwrite_newer_records(self, tmp_path):
        """Verify a second registry on the same file keeps the first one's writes."""
        path = tmp_path / "registry.json"
        first = Registry(path)
        second = Registry(path)
        first.add_graph(_graph("med"))
        second.add_graph(_graph("drug"))

        ids = {g.graph_id for g in Registry(path).list_graphs()}
        assert ids == {"med", "drug"}

    def test_returned_records_are_copies(self, tmp_path):
        """Verify editing a returned record does not change stored state."""
        reg = Registry(tmp_path / "registry.json")
        reg.add_graph(_graph())
        reg.get_graph("med").name = "edited"
        reg.list_graphs()[0].name = "edited"
        assert reg.get_graph("med").name is None

    def test_only_env_names_persisted(self, tmp_path):
        """Verify the on-disk file stores env-var names, not secret values."""
        path = tmp_path / "registry.json"
        reg = Registry(path)
        reg.add_graph(_graph())
        raw = json.loads(path.read_text("utf-8"))
        neo4j = raw["graphs"]["med"]["neo4j"]
        assert neo4j["user_env"] == "MED_NEO4J_USER"
        assert "password" not in neo4j
        assert neo4j["password_env"] == "MED_NEO4J_PASSWORD"

    def test_remove_graph_detaches_from_projects(self, tmp_path):
        """Verify removing a graph also removes it from project membership."""
        reg = Registry(tmp_path / "registry.json")
        reg.add_graph(_graph())
        reg.add_project(ProjectRecord(project_id="clinical", graph_ids=["med"]))
        reg.remove_graph("med")
        assert reg.get_project("clinical").graph_ids == []
        with pytest.raises(GraphNotFoundError):
            reg.get_graph("med")

    def test_update_graph_changes_name(self, tmp_path):
        """Verify update_graph applies a partial change and persists it."""
        reg = Registry(tmp_path / "registry.json")
        reg.add_graph(_graph())
        updated = reg.update_graph("med", name="Medical")
        assert updated.name == "Medical"
        assert Registry(reg.path).get_graph("med").name == "Medical"

    def test_project_crud(self, tmp_path):
        """Verify project add/get/remove and the not-found error."""
        reg = Registry(tmp_path / "registry.json")
        reg.add_project(ProjectRecord(project_id="p"))
        assert reg.get_project("p").project_id == "p"
        reg.remove_project("p")
        with pytest.raises(ProjectNotFoundError):
            reg.get_project("p")
