"""End-to-end ingestion write-path tests against real Neo4j and Qdrant.

These exercise only the write/identity/provenance layer of ``IngestionJob``: a
deterministic stub extractor (endpoints already set) is injected via the job's
``extractor`` seam so the model-backed ``ComposedExtractor`` never runs. They
verify that a text corpus produces entities and ``RELATES_TO`` edges, that
re-ingesting the same corpus is idempotent, and that a second corpus mentioning
the same entity unions its ``chunk_ids`` provenance instead of clobbering it.

Markers (``integration``/``enable_socket``) and skip-on-missing-config are
auto-applied by ``tests/conftest.py``; the ``/ingestion/`` path requires
``NEO4J_*`` and ``QDRANT_URL``.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.job import JobRecord
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.db.graph.neo4j import create_neo4j_client
from am_diag.db.vector.qdrant import create_qdrant_store
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.jobs import IngestionJob


_EMBED_DIM = 8
_COLLECTIONS = {
    "entity": "test_wp_entity",
    "relation": "test_wp_relation",
    "chunk": "test_wp_chunk",
    "community": "test_wp_community",
}


class _FixedEmbedder:
    """Deterministic embedder returning a fixed vector, with a stable model id."""

    model_id = "fixed-8d-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _EMBED_DIM for _ in texts]


class _CorpusSource:
    """One-batch corpus source yielding fixed documents."""

    source_kind = "corpus"

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    async def astream(self, batch_size: int = 100):
        yield self._documents


class _StubExtractor:
    """Emit a fixed entity pair and a fully-linked relation for each chunk."""

    def __init__(self, specs: list[tuple[str, str]], rel_type: str) -> None:
        self._specs = specs
        self._rel_type = rel_type

    async def extract(self, chunks) -> tuple[list[Entity], list[Relation]]:
        entities: list[Entity] = []
        relations: list[Relation] = []
        for chunk in chunks:
            ents = [
                Entity(
                    name=name,
                    label=label,
                    provenance=Provenance(
                        extractor=ExtractorName.LLM, chunk_ids=[chunk.id]
                    ),
                )
                for name, label in self._specs
            ]
            entities.extend(ents)
            if len(ents) >= 2:
                relations.append(
                    Relation(
                        head_id=ents[0].id,
                        tail_id=ents[1].id,
                        head_name=ents[0].name,
                        tail_name=ents[1].name,
                        type=self._rel_type,
                        provenance=Provenance(chunk_ids=[chunk.id]),
                    )
                )
        return entities, relations


def _job(tmp_path, target, cache, source, extractor) -> IngestionJob:
    return IngestionJob(
        JobRecord(job_id=uuid4().hex, kind="ingestion", graph_id="med"),
        target=target,
        cache=cache,
        state_dir=tmp_path,
        sources=[source],
        extractor=extractor,
    )


_ENTITY_COUNT = "MATCH (n:Entity) WHERE n.name ENDS WITH $run_id RETURN count(n) AS c"
_EDGE_COUNT = (
    "MATCH (a:Entity)-[r:RELATES_TO]->(:Entity) "
    "WHERE a.name ENDS WITH $run_id RETURN count(r) AS c"
)


async def _count(graph_store, query: str, params: dict | None = None) -> int:
    rows = await graph_store.execute_query(query, params)
    return rows[0]["c"] if rows else 0


async def _remove_run_data(graph_store, run_id: str) -> None:
    """Delete only the nodes this test run created, leaving other data alone."""
    await graph_store.execute_query(
        "MATCH (e:Entity) WHERE e.name ENDS WITH $run_id DETACH DELETE e",
        {"run_id": run_id},
    )
    await graph_store.execute_query(
        "MATCH (d:Document {source: $source}) "
        "OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d) DETACH DELETE c, d",
        {"source": f"inline-{run_id}"},
    )


@pytest.fixture
async def target():
    """Real Neo4j + Qdrant target with a fixed embedder and clean collections.

    Every entity name and document source carries a per-run id, so setup and
    teardown remove only this run's nodes and never touch other data in the
    configured database.
    """
    run_id = uuid4().hex[:8]
    graph_store = create_neo4j_client()
    vector_store = create_qdrant_store()
    await _remove_run_data(graph_store, run_id)
    for name in (
        _COLLECTIONS["entity"],
        _COLLECTIONS["relation"],
        _COLLECTIONS["chunk"],
    ):
        await vector_store.ensure_collection(name, _EMBED_DIM)
    yield SimpleNamespace(
        run_id=run_id,
        source=f"inline-{run_id}",
        specs=[
            (f"diabetes mellitus {run_id}", "Disease"),
            (f"metformin {run_id}", "Drug"),
        ],
        schema=MEDICAL_GRAPHRAG_SCHEMA,
        collections=_COLLECTIONS,
        graph_store=graph_store,
        vector_store=vector_store,
        embedder=_FixedEmbedder(),
    )
    await _remove_run_data(graph_store, run_id)
    for name in (
        _COLLECTIONS["entity"],
        _COLLECTIONS["relation"],
        _COLLECTIONS["chunk"],
    ):
        await vector_store.delete_collection(name)
    await graph_store.close()


class TestIngestionWritePath:
    """The write path produces edges and is idempotent + provenance-preserving."""

    async def test_single_ingest_produces_entities_and_edges(self, tmp_path, target):
        """A text corpus yields entities and at least one RELATES_TO edge."""
        docs = [Document(text="t1", source=target.source, external_id="1")]
        extractor = _StubExtractor(target.specs, "TREATED_BY")
        cache = IngestionCache(tmp_path / "med.db")
        await _job(tmp_path, target, cache, _CorpusSource(docs), extractor).run()

        entity_count = await _count(
            target.graph_store, _ENTITY_COUNT, {"run_id": target.run_id}
        )
        edge_count = await _count(
            target.graph_store, _EDGE_COUNT, {"run_id": target.run_id}
        )
        assert entity_count >= 2
        assert edge_count >= 1

    async def test_reingest_same_corpus_is_idempotent(self, tmp_path, target):
        """Ingesting the same corpus twice leaves node/edge counts unchanged."""
        docs = [Document(text="t1", source=target.source, external_id="1")]
        specs = target.specs
        cache = IngestionCache(tmp_path / "med.db")

        await _job(
            tmp_path,
            target,
            cache,
            _CorpusSource(docs),
            _StubExtractor(specs, "TREATED_BY"),
        ).run()
        entities_after_first = await _count(
            target.graph_store, _ENTITY_COUNT, {"run_id": target.run_id}
        )
        edges_after_first = await _count(
            target.graph_store, _EDGE_COUNT, {"run_id": target.run_id}
        )

        await _job(
            tmp_path,
            target,
            cache,
            _CorpusSource(docs),
            _StubExtractor(specs, "TREATED_BY"),
        ).run()
        entities_after_second = await _count(
            target.graph_store, _ENTITY_COUNT, {"run_id": target.run_id}
        )
        edges_after_second = await _count(
            target.graph_store, _EDGE_COUNT, {"run_id": target.run_id}
        )

        assert entities_after_second == entities_after_first
        assert edges_after_second == edges_after_first

    async def test_reingest_unions_provenance(self, tmp_path, target):
        """A shared entity across two corpora unions its chunk_ids provenance."""
        specs = target.specs
        cache = IngestionCache(tmp_path / "med.db")

        docs_a = [Document(text="t1", source=target.source, external_id="1")]
        docs_b = [Document(text="t2", source=target.source, external_id="2")]
        await _job(
            tmp_path,
            target,
            cache,
            _CorpusSource(docs_a),
            _StubExtractor(specs, "TREATED_BY"),
        ).run()
        await _job(
            tmp_path,
            target,
            cache,
            _CorpusSource(docs_b),
            _StubExtractor(specs, "TREATED_BY"),
        ).run()

        name = specs[0][0]
        node_count = await _count(
            target.graph_store,
            "MATCH (n:Entity {name: $name}) RETURN count(n) AS c",
            {"name": name},
        )
        rows = await target.graph_store.execute_query(
            "MATCH (n:Entity {name: $name}) RETURN n.chunk_ids AS chunk_ids",
            {"name": name},
        )
        assert node_count == 1
        assert rows and len(rows[0]["chunk_ids"]) == 2
