"""The ``Project`` handle: an owned, shareable container of graphs.

A project is the agent's federation boundary and the unit of sharing. It manages
graph membership, federated search across its graphs, and the agent surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from am_diag.agents.events import map_event
from am_diag.agents.settings import AgentSettings
from am_diag.api.agent_runtime import (
    agent_config_hash,
    extract_clinical_answer,
    to_messages,
)
from am_diag.api.context import pack_context
from am_diag.api.errors import GraphNotFoundError
from am_diag.api.federation import FederatedSearch, federated_search
from am_diag.api.graph import Graph
from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.agent_events import AgentEvent
from am_diag.common.data_models.message import Message
from am_diag.common.data_models.project import ProjectRecord
from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.config import SearchConfig
from am_diag.retrieval.filters import SearchFilters


if TYPE_CHECKING:
    from am_diag.api.connection_pool import ConnectionPool
    from am_diag.api.registry import Registry


class Project:
    """A federation/agent boundary over one or more graphs.

    Construct an ad hoc project over live graphs with ``Project(graphs)``; obtain
    a Workspace-managed project with ``Workspace.project(id)``.

    Args:
        graphs: Live ``Graph`` objects to federate over.
        project_id: Optional label for the ad hoc project.
    """

    def __init__(
        self,
        graphs: list[Graph] | None = None,
        project_id: str = "ad-hoc",
        *,
        registry: Registry | None = None,
        pool: ConnectionPool | None = None,
    ) -> None:
        """Bind a project to live graphs (ad hoc) or a registry + pool (managed).

        Use ``Project(graphs)`` for an ad hoc project over a fixed set of live
        graphs. ``Workspace.project(id)`` builds the managed form, passing the
        ``registry`` and ``pool`` keyword arguments.
        """
        self._project_id = project_id
        self._graphs: list[Graph] | None = list(graphs) if graphs is not None else None
        self._registry = registry
        self._pool = pool
        self._agents: dict[tuple[str, tuple[str, ...]], object] = {}

    @property
    def project_id(self) -> str:
        """Id of the target project."""
        return self._project_id

    def _record(self) -> ProjectRecord:
        """Return the persisted (ad hoc: synthesized) record for this project."""
        if self._registry is None:
            return ProjectRecord(
                project_id=self._project_id,
                graph_ids=[g.graph_id for g in self._graphs or []],
            )
        return self._registry.get_project(self._project_id)

    async def graphs(self) -> list[str]:
        """Return the ids of the graphs in this project."""
        return [g.graph_id for g in self._live_graphs(None)]

    def _live_graphs(self, graphs: list[str] | None) -> list[Graph]:
        """Resolve target live graphs, validating project membership."""
        if self._pool is None:
            live = list(self._graphs or [])
        else:
            live = [self._pool.graph(gid) for gid in self._record().graph_ids]
        if graphs is None:
            return live
        by_id = {graph.graph_id: graph for graph in live}
        missing = [gid for gid in graphs if gid not in by_id]
        if missing:
            raise GraphNotFoundError(
                f"Graph {missing[0]!r} is not a member of project {self._project_id!r}"
            )
        return [by_id[gid] for gid in graphs]

    def _target_graphs(self, graphs: list[str] | None) -> list[str]:
        """Resolve the target graph set, validating project membership.

        Args:
            graphs: Optional subset of graph ids; ``None`` means all members.

        Raises:
            GraphNotFoundError: If a requested graph is not a project member.
        """
        return [graph.graph_id for graph in self._live_graphs(graphs)]

    async def search(
        self,
        query: str,
        *,
        graphs: list[str] | None = None,
        recipe: SearchConfig | str = "hybrid_rrf",
        filters: SearchFilters | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Run a federated search across the project's graphs.

        Args:
            query: Natural-language query.
            graphs: Optional subset of member graphs to search.
            recipe: Recipe name or ``SearchConfig`` applied per graph.
            filters: Optional metadata filters.
            limit: Maximum fused results returned.

        Returns:
            Fused results tagged with their ``source_graph_id``.
        """
        return await federated_search(
            self._live_graphs(graphs),
            query,
            recipe=recipe,
            filters=filters,
            limit=limit,
        )

    async def context(
        self,
        query: str,
        *,
        graphs: list[str] | None = None,
        recipe: SearchConfig | str = "hybrid_rrf",
        filters: SearchFilters | None = None,
        max_characters: int = 2500,
    ) -> str:
        """Return a packed context block from a federated search."""
        results = await self.search(
            query, graphs=graphs, recipe=recipe, filters=filters
        )
        return pack_context(results, max_characters)

    def _resolved_settings(self, override: AgentSettings | None) -> AgentSettings:
        """Merge per-call, then per-project, then env-default agent settings."""
        if override is not None:
            return override
        return AgentSettings(**(self._record().agent_settings or {}))

    async def _agent(
        self, graphs: list[str] | None, settings: AgentSettings | None
    ) -> Any:
        """Build (and cache per graph-set) the agent wired to federated tools."""
        from am_diag.agents.agent import build_clinical_agent  # noqa: PLC0415

        target = tuple(sorted(self._target_graphs(graphs)))
        resolved = self._resolved_settings(settings)
        cache = self._agents
        cache_key = (self._project_id, target)
        # Per-call settings bypass the shared cache so they never pollute it.
        if settings is None and cache_key in cache:
            return cache[cache_key]
        fed = FederatedSearch(self._live_graphs(list(target)))
        agent = await build_clinical_agent(fed, resolved)
        if settings is None:
            cache[cache_key] = agent
        return agent

    async def answer(
        self,
        query_or_messages: str | list[Message],
        *,
        graphs: list[str] | None = None,
        settings: AgentSettings | None = None,
    ) -> ClinicalAnswer:
        """Answer a question (or multi-turn conversation) over the project.

        Args:
            query_or_messages: A question string or a list of ``Message`` turns.
            graphs: Optional subset of member graphs to federate over.
            settings: Optional per-call agent settings override.

        Returns:
            The structured clinical answer.
        """
        agent = await self._agent(graphs, settings)
        resolved = self._resolved_settings(settings)
        out = await agent.ainvoke(
            {"messages": to_messages(query_or_messages)},
            config={"recursion_limit": resolved.recursion_limit},
        )
        return extract_clinical_answer(out)

    async def stream(
        self,
        query_or_messages: str | list[Message],
        *,
        graphs: list[str] | None = None,
        settings: AgentSettings | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Stream typed agent events while answering.

        Yields:
            :data:`AgentEvent` values (tool calls, tokens, ...) as they occur.
        """
        agent = await self._agent(graphs, settings)
        async for event in agent.astream_events(
            {"messages": to_messages(query_or_messages)}, version="v2"
        ):
            mapped = map_event(event)
            if mapped is not None:
                yield mapped

    async def answer_batch(
        self,
        questions: list[str],
        *,
        graphs: list[str] | None = None,
        settings: AgentSettings | None = None,
        config: Any | None = None,
    ) -> Any:
        """Create a durable QA job over a batch of questions (does not run).

        Returns:
            A :class:`QAJob`; call ``run()`` to evaluate and cache answers.
        """
        from am_diag.common.data_models.job import JobConfig, JobRecord  # noqa: PLC0415
        from am_diag.ingestion.jobs import QAJob, new_job_id  # noqa: PLC0415

        target = self._target_graphs(graphs)
        if not target:
            raise GraphNotFoundError(
                f"Project {self._project_id!r} has no graphs to answer over"
            )
        resolved = self._resolved_settings(settings)
        target_graph = self._live_graphs([target[0]])[0]
        record = JobRecord(
            job_id=new_job_id(),
            kind="qa",
            project_id=self._project_id,
            graphs=target,
            config=config or JobConfig(),
        )
        return QAJob(
            record,
            answerer=self,
            cache=target_graph.ensure_cache(),
            state_dir=target_graph.state_dir,
            questions=list(questions),
            agent_config_hash=agent_config_hash(resolved, target),
        )

    async def add_graph(self, graph_id: str) -> None:
        """Add a graph to the project.

        Raises:
            GraphNotFoundError: If the graph is not registered.
        """
        if self._registry is None:
            raise TypeError("add_graph() requires a Workspace-backed Project")
        self._registry.get_graph(graph_id)
        record = self._record()
        if graph_id in record.graph_ids:
            return
        graph_ids = [*record.graph_ids, graph_id]
        self._registry.update_project(self._project_id, graph_ids=graph_ids)

    async def remove_graph(self, graph_id: str) -> None:
        """Remove a graph from the project.

        Raises:
            GraphNotFoundError: If the graph is not a member of the project.
        """
        record = self._record()
        if graph_id not in record.graph_ids:
            raise GraphNotFoundError(
                f"Graph {graph_id!r} is not a member of project {self._project_id!r}"
            )
        if self._registry is None:
            raise TypeError("remove_graph() requires a Workspace-backed Project")
        graph_ids = [g for g in record.graph_ids if g != graph_id]
        self._registry.update_project(self._project_id, graph_ids=graph_ids)
