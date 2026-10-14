"""Workspace-managed registry, connection pool, and job namespace."""

from __future__ import annotations

import os
from pathlib import Path

from am_diag.api.admin import GraphAdmin, ProjectAdmin
from am_diag.api.connection_pool import ConnectionPool
from am_diag.api.graph import Graph
from am_diag.api.project import Project
from am_diag.api.registry import Registry
from am_diag.ingestion.cache import CacheManager
from am_diag.ingestion.jobs import IngestionJob, JobManager


_STATE_DIR_ENV = "AM_DIAG_STATE_DIR"
_DEFAULT_STATE_DIR = Path(".am_diag")


class Workspace:
    """Managed registry + connection pool for named graphs and projects.

    Construct via :meth:`from_env` or :meth:`open`; use as an async context
    manager so cached graphs are closed on exit.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        state_dir: Path,
        cache_manager: CacheManager | None = None,
        live_jobs: dict[str, IngestionJob] | None = None,
    ) -> None:
        """Initialize the composition root around a registry and state directory.

        Args:
            registry: The opened registry.
            state_dir: Root directory for caches and durable job state.
            cache_manager: Content-cache manager. Defaults to one rooted at
                ``state_dir/cache``; injected by tests sharing a cache with a
                prebuilt graph.
            live_jobs: In-session job registry. Defaults to a fresh dict; injected
                by tests so a prebuilt graph and this workspace share one registry.
        """
        self._registry = registry
        self._state_dir = state_dir
        self._cache = cache_manager or CacheManager(state_dir / "cache")
        self._live_jobs: dict[str, IngestionJob] = (
            live_jobs if live_jobs is not None else {}
        )
        self._pool = ConnectionPool(
            registry,
            state_dir=state_dir,
            cache_manager=self._cache,
            live_jobs=self._live_jobs,
        )
        self._graphs = GraphAdmin(registry, self._pool)
        self._projects = ProjectAdmin(registry)
        self._projects_cache: dict[str, Project] = {}
        self._job_manager = JobManager(state_dir, self._live_jobs)

    @classmethod
    def from_env(cls) -> Workspace:
        """Open the client using environment-configured paths."""
        return cls.open()

    @classmethod
    def open(
        cls,
        registry_path: Path | None = None,
        *,
        state_dir: Path | None = None,
    ) -> Workspace:
        """Open the client, creating the registry and state directory paths.

        Args:
            registry_path: Explicit registry file path. Defaults to the
                ``AM_DIAG_REGISTRY_PATH`` env var, else ``<state_dir>/registry.json``.
            state_dir: Root state directory. Defaults to ``AM_DIAG_STATE_DIR``
                env var, else ``.am_diag``.
        """
        if state_dir is None:
            env_dir = os.environ.get(_STATE_DIR_ENV)
            state_dir = Path(env_dir) if env_dir else _DEFAULT_STATE_DIR
        if registry_path is None and not os.environ.get("AM_DIAG_REGISTRY_PATH"):
            registry_path = state_dir / "registry.json"
        registry = Registry.open(registry_path)
        return cls(registry, state_dir=state_dir)

    @property
    def graphs(self) -> GraphAdmin:
        """Admin namespace for graph deployments."""
        return self._graphs

    @property
    def projects(self) -> ProjectAdmin:
        """Admin namespace for projects."""
        return self._projects

    @property
    def jobs(self) -> JobManager:
        """Manager for durable ingestion/QA jobs."""
        return self._job_manager

    def graph(self, graph_id: str) -> Graph:
        """Return the named live graph, building it from the registry if needed.

        Delegates to the connection pool, which builds once and caches. Building
        resolves the record's env-ref secrets eagerly, so a missing secret raises
        ``ConnectionError`` here rather than at first use.
        """
        return self._pool.graph(graph_id)

    def project(self, project_id: str) -> Project:
        """Return the (cached) managed handle to ``project_id``.

        The same instance is returned across calls so its agent cache persists;
        membership and settings are read fresh from the registry each call.
        """
        cached = self._projects_cache.get(project_id)
        if cached is None:
            cached = Project(
                project_id=project_id, registry=self._registry, pool=self._pool
            )
            self._projects_cache[project_id] = cached
        return cached

    def attach_graph(self, graph: Graph) -> None:
        """Register an already-built live graph (bring-your-own) into the pool.

        Makes a ``Graph`` built via :meth:`Graph.connect` addressable by
        ``graph(id)`` without a registry record. The pool then owns the attached
        graph's close lifecycle (see :meth:`ConnectionPool.attach`).
        """
        self._pool.attach(graph)

    async def aclose(self) -> None:
        """Close every cached graph and release resources."""
        await self._pool.close()
        await self._cache.close()

    async def __aenter__(self) -> Workspace:
        """Enter the async context, returning self."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit the async context, closing all graphs."""
        await self.aclose()
