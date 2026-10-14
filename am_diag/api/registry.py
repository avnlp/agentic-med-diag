"""File-backed registry of Project and Graph records.

The registry is the control plane: a single JSON file mapping ids to records.
Reads are served from an in-memory cache; every mutation rewrites the file
atomically (temp file + ``os.replace``) so a crash mid-write cannot corrupt it.
Only env-var names are stored — never secret values.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from am_diag.api.errors import GraphNotFoundError, ProjectNotFoundError, RegistryError
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.project import ProjectRecord


_REGISTRY_VERSION = 1
_REGISTRY_PATH_ENV = "AM_DIAG_REGISTRY_PATH"
_DEFAULT_REGISTRY_PATH = Path(".am_diag") / "registry.json"


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Registry:
    """File-backed store of :class:`ProjectRecord` and :class:`GraphRecord`.

    On-disk shape::

        {"version": 1, "graphs": {gid: GraphRecord}, "projects": {pid: ...}}

    Reads are cached in memory; every mutation rewrites the file atomically.
    Secrets are never stored — only env-var names.
    """

    def __init__(self, path: Path) -> None:
        """Open (or initialize) the registry at ``path``.

        Args:
            path: Location of the registry JSON file. Parent directories are
                created on first write.
        """
        self._path = path
        self._graphs: dict[str, GraphRecord] = {}
        self._projects: dict[str, ProjectRecord] = {}
        self._load()

    @classmethod
    def open(cls, path: Path | None = None) -> Registry:
        """Open the registry at ``path`` or the configured default.

        Args:
            path: Explicit path. When ``None``, uses ``AM_DIAG_REGISTRY_PATH``
                from the environment, else ``.am_diag/registry.json``.
        """
        if path is None:
            env_path = os.environ.get(_REGISTRY_PATH_ENV)
            path = Path(env_path) if env_path else _DEFAULT_REGISTRY_PATH
        return cls(path)

    @property
    def path(self) -> Path:
        """Filesystem path of the registry file."""
        return self._path

    def _load(self) -> None:
        """Populate the in-memory caches from disk, if the file exists."""
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text("utf-8"))
        self._graphs = {
            gid: GraphRecord.model_validate(rec)
            for gid, rec in raw.get("graphs", {}).items()
        }
        self._projects = {
            pid: ProjectRecord.model_validate(rec)
            for pid, rec in raw.get("projects", {}).items()
        }

    def _save(self) -> None:
        """Atomically rewrite the registry file from the in-memory caches."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _REGISTRY_VERSION,
            "graphs": {
                gid: rec.model_dump(mode="json") for gid, rec in self._graphs.items()
            },
            "projects": {
                pid: rec.model_dump(mode="json") for pid, rec in self._projects.items()
            },
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), "utf-8")
        os.replace(tmp, self._path)

    def add_graph(self, rec: GraphRecord) -> None:
        """Persist a new graph record.

        Raises:
            RegistryError: If a graph with the same id already exists.
        """
        if rec.graph_id in self._graphs:
            raise RegistryError(f"Graph already exists: {rec.graph_id!r}")
        self._graphs[rec.graph_id] = rec
        self._save()

    def get_graph(self, graph_id: str) -> GraphRecord:
        """Return the graph record for ``graph_id``.

        Raises:
            GraphNotFoundError: If no such graph exists.
        """
        rec = self._graphs.get(graph_id)
        if rec is None:
            raise GraphNotFoundError(f"No such graph: {graph_id!r}")
        return rec

    def list_graphs(self) -> list[GraphRecord]:
        """Return all graph records."""
        return list(self._graphs.values())

    def update_graph(self, graph_id: str, **changes: Any) -> GraphRecord:
        """Apply ``changes`` to a graph record and persist.

        Raises:
            GraphNotFoundError: If no such graph exists.
        """
        rec = self.get_graph(graph_id)
        data = rec.model_dump()
        data.update({k: v for k, v in changes.items() if v is not None})
        data["updated_at"] = _utcnow()
        updated = GraphRecord.model_validate(data)
        self._graphs[graph_id] = updated
        self._save()
        return updated

    def remove_graph(self, graph_id: str) -> None:
        """Delete a graph record and remove it from every project.

        Raises:
            GraphNotFoundError: If no such graph exists.
        """
        if graph_id not in self._graphs:
            raise GraphNotFoundError(f"No such graph: {graph_id!r}")
        del self._graphs[graph_id]
        for pid, proj in self._projects.items():
            if graph_id in proj.graph_ids:
                proj.graph_ids = [g for g in proj.graph_ids if g != graph_id]
                proj.updated_at = _utcnow()
                self._projects[pid] = proj
        self._save()

    def add_project(self, rec: ProjectRecord) -> None:
        """Persist a new project record.

        Raises:
            RegistryError: If a project with the same id already exists.
        """
        if rec.project_id in self._projects:
            raise RegistryError(f"Project already exists: {rec.project_id!r}")
        self._projects[rec.project_id] = rec
        self._save()

    def get_project(self, project_id: str) -> ProjectRecord:
        """Return the project record for ``project_id``.

        Raises:
            ProjectNotFoundError: If no such project exists.
        """
        rec = self._projects.get(project_id)
        if rec is None:
            raise ProjectNotFoundError(f"No such project: {project_id!r}")
        return rec

    def list_projects(self) -> list[ProjectRecord]:
        """Return all project records."""
        return list(self._projects.values())

    def update_project(self, project_id: str, **changes: Any) -> ProjectRecord:
        """Apply ``changes`` to a project record and persist.

        Raises:
            ProjectNotFoundError: If no such project exists.
        """
        rec = self.get_project(project_id)
        data = rec.model_dump()
        data.update({k: v for k, v in changes.items() if v is not None})
        data["updated_at"] = _utcnow()
        updated = ProjectRecord.model_validate(data)
        self._projects[project_id] = updated
        self._save()
        return updated

    def remove_project(self, project_id: str) -> None:
        """Delete a project record.

        Raises:
            ProjectNotFoundError: If no such project exists.
        """
        if project_id not in self._projects:
            raise ProjectNotFoundError(f"No such project: {project_id!r}")
        del self._projects[project_id]
        self._save()
