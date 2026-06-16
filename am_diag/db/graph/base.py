"""Abstract base class for graph database interactions.

Defines the ``GraphStore`` ABC that all graph backend clients must implement.
The ABC covers both low-level transport primitives (``execute_query``,
``execute_batch``) and the full granular ingest API (upsert_documents …
write_knowledge_graph). Concrete implementations (e.g. ``Neo4jClient``) load
all Cypher from ``am_diag.common.cypher.loader`` rather than embedding query
strings here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema


class GraphStore(ABC):
    """Abstract base for graph database clients (Neo4j, etc.).

    Subclasses must implement the transport primitives (``execute_query``,
    ``execute_batch``, ``close``, ``transaction``, ``health_check``) and all
    granular ingest methods. ``write_knowledge_graph`` is declared abstract
    here so that concrete clients control the ordering guarantee
    (constraints → nodes → relationships).
    """

    @abstractmethod
    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return records as plain dicts.

        Args:
            query: Cypher string with ``$name`` placeholders.
            params: Optional parameter bindings.

        Returns:
            Result records as ``list[dict[str, Any]]``. Empty list when the
            query produces no rows.
        """

    @abstractmethod
    async def execute_batch(
        self,
        query: str,
        records: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        """Execute ``query`` for each chunk of ``records``.

        Splits ``records`` into slices of ``batch_size`` and calls
        ``execute_query`` once per slice, binding the slice to ``$records``.

        Args:
            query: Cypher with ``UNWIND $records AS r`` semantics.
            records: Row data to pass as the ``$records`` parameter.
            batch_size: Maximum records per driver call.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the graph database connection and release resources."""

    @abstractmethod
    def transaction(self) -> Any:
        """Return an async context manager for an atomic read-write transaction.

        Returns:
            An async context manager that commits on clean exit and rolls back
            on any exception.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connectivity to the graph backend.

        Returns:
            ``True`` if the backend is reachable.

        Raises:
            Exception: If the backend is unreachable.
        """

    @abstractmethod
    async def setup_schema(self, schema: GraphSchema) -> None:
        """Create constraints and indexes for all types in ``schema``.

        Implementations should be idempotent (IF NOT EXISTS / IF NOT EXISTS).

        Args:
            schema: Entity and relation type definitions that drive DDL generation.
        """

    @abstractmethod
    async def upsert_documents(self, docs: list[Document]) -> None:
        """Insert or update Document nodes.

        Args:
            docs: Documents to write; empty list is a no-op.
        """

    @abstractmethod
    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or update Chunk nodes.

        Args:
            chunks: Chunks to write; empty list is a no-op.
        """

    @abstractmethod
    async def upsert_entities(self, entities: list[Entity]) -> None:
        """Insert or update Entity nodes, applying secondary type labels.

        Implementations must group entities by ``entity.label`` and issue
        separate queries per group so that Neo4j's secondary-label MERGE is
        applied correctly.

        Args:
            entities: Entities to write; empty list is a no-op.
        """

    @abstractmethod
    async def upsert_relations(self, relations: list[Relation]) -> None:
        """Insert or update Relation edges between Entity nodes.

        Only relations with non-None ``head_id`` and ``tail_id`` are written;
        unresolved relations are silently skipped.

        Args:
            relations: Relations to write.
        """

    @abstractmethod
    async def link_part_of(self, chunks: list[Chunk]) -> None:
        """Create PART_OF edges from each Chunk to its parent Document.

        Args:
            chunks: Chunks whose document_id links should be materialized.
        """

    @abstractmethod
    async def link_next_chunk(self, document_ids: list[str]) -> None:
        """Create NEXT_CHUNK edges between consecutive chunks in a document.

        Called once per document after all chunks have been upserted.

        Args:
            document_ids: Document UUIDs (as strings) to sequence chunks for.
        """

    @abstractmethod
    async def link_has_entity(self, entities: list[Entity]) -> None:
        """Create HAS_ENTITY edges from Chunk nodes to Entity nodes.

        Args:
            entities: Entities with populated ``provenance.chunk_ids``.
        """

    @abstractmethod
    async def upsert_communities(self, communities: list[Community]) -> None:
        """Insert or update Community nodes.

        Args:
            communities: Communities to write; empty list is a no-op.
        """

    @abstractmethod
    async def link_in_community(self, communities: list[Community]) -> None:
        """Create IN_COMMUNITY edges from Entity nodes to Community nodes.

        Args:
            communities: Communities whose ``entity_ids`` should be linked.
        """

    @abstractmethod
    async def link_parent_community(self, communities: list[Community]) -> None:
        """Create PARENT_COMMUNITY edges between nested Community nodes.

        Communities with ``parent is None`` are silently skipped.

        Args:
            communities: Communities to inspect for parent links.
        """

    @abstractmethod
    async def upsert_reports(self, reports: list[CommunityReport]) -> None:
        """Insert or update CommunityReport nodes.

        Args:
            reports: Reports to write; empty list is a no-op.
        """

    @abstractmethod
    async def link_has_report(self, reports: list[CommunityReport]) -> None:
        """Create HAS_REPORT edges from Community nodes to CommunityReport nodes.

        Args:
            reports: Reports whose ``community_id`` links should be materialized.
        """

    @abstractmethod
    async def write_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """Orchestrate a full KnowledgeGraph write in dependency order.

        Implementations must enforce the ordering guarantee:
        1. Schema constraints and indexes (DDL)
        2. Nodes (Document → Chunk → Entity → Community → CommunityReport)
        3. Relationships (PART_OF, NEXT_CHUNK, HAS_ENTITY, RELATES_TO,
           IN_COMMUNITY, PARENT_COMMUNITY, HAS_REPORT)

        Args:
            kg: The assembled KnowledgeGraph to persist.
        """


__all__ = ["GraphStore"]
