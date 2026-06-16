"""Concrete Neo4j client implementation.

Loads all Cypher from ``am_diag.common.cypher.loader`` (``load`` for static
templates, ``render`` for templates with dynamic identifier placeholders such
as entity type labels). No Cypher string literals appear in this file.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from neo4j import AsyncGraphDatabase
from typing_extensions import LiteralString

from am_diag.common.cypher._identifiers import validate_label
from am_diag.common.cypher.loader import load, render
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.knowledge_graph import KnowledgeGraph
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.serialize import (
    chunk_record,
    community_record,
    document_record,
    entity_record,
    has_entity_records,
    has_report_records,
    in_community_records,
    parent_community_records,
    relation_record,
    report_record,
)
from am_diag.db.graph.settings import Neo4jSettings


logger = logging.getLogger(__name__)


class Neo4jTransaction:
    """Wraps a neo4j ``AsyncTransaction``, exposing a simple ``run`` interface.

    Instances are yielded by ``Neo4jClient.transaction`` and should not
    be constructed directly.
    """

    def __init__(self, tx: Any) -> None:
        """Wrap a neo4j AsyncTransaction.

        Args:
            tx: Neo4j AsyncTransaction instance.
        """
        self._tx = tx

    async def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Run a Cypher statement within this transaction.

        Args:
            cypher: Cypher query string with ``$name`` placeholders.
            **params: Keyword arguments bound to Cypher parameters.

        Returns:
            Result records as ``list[dict[str, Any]]``.
        """
        result = await self._tx.run(cypher, **params)
        return await result.data()


class Neo4jClient(GraphStore):
    """Async Neo4j client wrapping the official ``neo4j`` driver.

    All Cypher is loaded from ``am_diag.common.cypher.loader``; no query
    strings are embedded in this class. Use ``create_neo4j_client`` as the
    preferred factory when settings should be read from the environment.

    Args:
        settings: Connection settings (URI, user, password, database).
    """

    def __init__(self, settings: Neo4jSettings) -> None:
        """Initialise the Neo4j client and create a driver connection.

        Args:
            settings: Connection settings (URI, user, password, database).
        """
        self._driver = AsyncGraphDatabase.driver(
            settings.uri,
            auth=(settings.user, settings.password),
        )
        self._database = settings.database

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return records as plain dicts.

        Suitable for both write and read operations. Parameters are passed
        server-side to prevent injection.

        Args:
            query: Cypher query string with ``$name`` placeholders.
            params: Optional parameter bindings for ``$name`` placeholders.

        Returns:
            Result records as ``list[dict[str, Any]]``. Empty list when the
            query returns no rows.

        Raises:
            neo4j.exceptions.Neo4jError: On query or connectivity failure.
        """
        result = await self._driver.execute_query(
            cast(LiteralString, query),
            parameters_=params,
            database_=self._database,
        )
        return [r.data() for r in result.records]

    async def execute_batch(
        self,
        query: str,
        records: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        """Execute ``query`` for each chunk of ``records``.

        Each slice is sent in an explicit transaction so that a failure in one
        batch does not silently corrupt others.

        Args:
            query: Cypher with an unwind clause and ``$records`` parameter.
            records: Row data to bind to ``$records``.
            batch_size: Maximum records per driver call.
        """
        if not records:
            return
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            async with self.transaction():
                await self.execute_query(query, {"records": batch})

    async def close(self) -> None:
        """Close the underlying driver and release connections."""
        await self._driver.close()

    async def __aenter__(self) -> "Neo4jClient":
        """Enter async context manager, returning self."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager, closing the driver."""
        await self.close()

    async def health_check(self) -> bool:
        """Verify connectivity to Neo4j.

        Returns:
            ``True`` when the instance is reachable.

        Raises:
            Exception: If the Neo4j instance is unreachable.
        """
        await self._driver.verify_connectivity()
        return True

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Neo4jTransaction]:
        """Context manager for explicit, atomic transactions.

        Opens a driver session, begins a transaction, and commits on normal
        exit. Rolls back and re-raises on any exception.

        Yields:
            A ``Neo4jTransaction`` handle for running Cypher statements.

        Raises:
            neo4j.exceptions.Neo4jError: On driver or Cypher failure.
        """
        async with self._driver.session(database=self._database) as session:
            tx = await session.begin_transaction()
            try:
                yield Neo4jTransaction(tx)
                await tx.commit()
            except BaseException:
                await tx.rollback()
                raise

    async def setup_schema(self, schema: GraphSchema) -> None:
        """Create constraints for each entity label, then shared indexes.

        One uniqueness constraint is created per entity label plus one each
        for Document, Chunk, Community, and CommunityReport. All statements
        use ``IF NOT EXISTS`` so repeated calls are idempotent.

        Args:
            schema: Entity and relation type definitions that drive DDL.
        """
        all_labels = [et.label for et in schema.entity_types] + [
            "Document",
            "Chunk",
            "Community",
            "CommunityReport",
        ]
        for label in all_labels:
            validate_label(label)
            cypher = render("schema/constraints", Label=label)
            await self.execute_query(cypher, {})
        await self.execute_query(load("schema/indexes"), {})

    async def upsert_documents(self, docs: list[Document]) -> None:
        """Insert or update Document nodes via upsert_documents.cypher.

        Args:
            docs: Documents to write; empty list is a no-op.
        """
        if not docs:
            return
        records = [document_record(d) for d in docs]
        await self.execute_batch(load("ingest/upsert_documents"), records)

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert or update Chunk nodes via upsert_chunks.cypher.

        Args:
            chunks: Chunks to write; empty list is a no-op.
        """
        if not chunks:
            return
        records = [chunk_record(c) for c in chunks]
        await self.execute_batch(load("ingest/upsert_chunks"), records)

    async def upsert_entities(self, entities: list[Entity]) -> None:
        """Group entities by label and upsert each group via a rendered template.

        Each entity label requires its own rendered Cypher because the secondary
        Neo4j label (e.g. ``:Drug``) is a Cypher identifier, not a parameter.
        Grouping before rendering avoids N round-trips to the loader cache.

        Args:
            entities: Entities to write; empty list is a no-op.
        """
        if not entities:
            return
        by_label: dict[str, list[dict[str, Any]]] = {}
        for e in entities:
            by_label.setdefault(e.label, []).append(entity_record(e))
        for label, records in by_label.items():
            cypher = render("ingest/upsert_entities", Label=label)
            await self.execute_batch(cypher, records)

    async def upsert_relations(self, relations: list[Relation]) -> None:
        """Insert or update Relation edges for resolved head/tail pairs.

        Relations without ``head_id`` or ``tail_id`` are skipped because the
        template's node lookup would find no nodes and produce no edges.

        Args:
            relations: Relations to write; unresolved ones are silently skipped.
        """
        resolved = [r for r in relations if r.head_id and r.tail_id]
        if not resolved:
            return
        records = [relation_record(r) for r in resolved]
        await self.execute_batch(load("ingest/upsert_relations"), records)

    async def link_part_of(self, chunks: list[Chunk]) -> None:
        """Create PART_OF edges from each Chunk to its parent Document.

        Args:
            chunks: Chunks to link; empty list is a no-op.
        """
        if not chunks:
            return
        records = [{"id": str(c.id), "document_id": str(c.document_id)} for c in chunks]
        await self.execute_batch(load("ingest/link_part_of"), records)

    async def link_next_chunk(self, document_ids: list[str]) -> None:
        """Create NEXT_CHUNK edges between consecutive chunks in each document.

        One query per document is issued; the Cypher template orders by
        chunk_index and merges edges between adjacent nodes.

        Args:
            document_ids: Document UUID strings to sequence chunks for.
        """
        for doc_id in document_ids:
            await self.execute_query(
                load("ingest/link_next_chunk"),
                {"document_id": doc_id},
            )

    async def link_has_entity(self, entities: list[Entity]) -> None:
        """Create HAS_ENTITY edges from Chunk nodes to Entity nodes.

        Args:
            entities: Entities with populated ``provenance.chunk_ids``.
        """
        records = has_entity_records(entities)
        if records:
            await self.execute_batch(load("ingest/link_has_entity"), records)

    async def upsert_communities(self, communities: list[Community]) -> None:
        """Insert or update Community nodes via upsert_communities.cypher.

        Args:
            communities: Communities to write; empty list is a no-op.
        """
        if not communities:
            return
        records = [community_record(c) for c in communities]
        await self.execute_batch(load("community/upsert_communities"), records)

    async def link_in_community(self, communities: list[Community]) -> None:
        """Create IN_COMMUNITY edges from Entity nodes to Community nodes.

        Args:
            communities: Communities whose ``entity_ids`` should be linked.
        """
        records = in_community_records(communities)
        if records:
            await self.execute_batch(load("community/link_in_community"), records)

    async def link_parent_community(self, communities: list[Community]) -> None:
        """Create PARENT_COMMUNITY edges between nested Community nodes.

        Communities with ``parent is None`` produce no records and the call
        becomes a no-op.

        Args:
            communities: Communities to inspect for parent links.
        """
        records = parent_community_records(communities)
        if records:
            await self.execute_batch(load("community/link_parent_community"), records)

    async def upsert_reports(self, reports: list[CommunityReport]) -> None:
        """Insert or update CommunityReport nodes via upsert_reports.cypher.

        Args:
            reports: Reports to write; empty list is a no-op.
        """
        if not reports:
            return
        records = [report_record(r) for r in reports]
        await self.execute_batch(load("community/upsert_reports"), records)

    async def link_has_report(self, reports: list[CommunityReport]) -> None:
        """Create HAS_REPORT edges from Community nodes to CommunityReport nodes.

        Args:
            reports: Reports whose ``community_id`` links should be materialized.
        """
        records = has_report_records(reports)
        if records:
            await self.execute_batch(load("community/link_has_report"), records)

    async def write_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """Orchestrate a full KnowledgeGraph write in dependency order.

        The ordering guarantee — constraints → nodes → relationships — ensures
        that upsert statements on relationships always find their endpoint nodes
        and that uniqueness constraints are in place before any node is written.

        Args:
            kg: The assembled KnowledgeGraph to persist.
        """
        if kg.graph_schema:
            await self.setup_schema(kg.graph_schema)
        await self.upsert_documents(kg.documents)
        await self.upsert_chunks(kg.chunks)
        await self.upsert_entities(kg.entities)
        await self.upsert_communities(kg.communities)
        await self.upsert_reports(kg.reports)
        # Relationships after all nodes so node lookups always succeed.
        await self.link_part_of(kg.chunks)
        doc_ids = list({str(c.document_id) for c in kg.chunks})
        await self.link_next_chunk(doc_ids)
        await self.link_has_entity(kg.entities)
        await self.upsert_relations(kg.relations)
        await self.link_in_community(kg.communities)
        await self.link_parent_community(kg.communities)
        await self.link_has_report(kg.reports)


def create_neo4j_client(settings: Neo4jSettings | None = None) -> Neo4jClient:
    """Instantiate a ``Neo4jClient`` from settings.

    Args:
        settings: Explicit settings object. When ``None``, settings are
            read from the environment (``NEO4J_URI``, ``NEO4J_USER``,
            ``NEO4J_PASSWORD``, ``NEO4J_DATABASE``).

    Returns:
        A configured ``Neo4jClient`` ready for use.
    """
    return Neo4jClient(settings or Neo4jSettings())
