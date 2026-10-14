"""Tests for the durable IngestionJob engine.

Drives the corpus path with an injected fake extractor and chunker against a
real temp SQLite cache and a mocked graph/vector target. Verifies item
lifecycle, poison-item isolation (partial + failures), resume skipping, and that
extraction/embedding cache hits skip the extractor and embedder.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.job import JobRecord
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.jobs import IngestionJob


class _FakeCorpusSource:
    """One-batch corpus source yielding fixed documents."""

    source_kind = "corpus"

    def __init__(self, documents):
        self._documents = documents

    async def astream(self, batch_size=100):
        yield self._documents


class _FakeExtractor:
    """Wrap a per-chunk async fn as an ``Extractor`` for the job seam."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    async def extract(self, chunks):
        entities, relations = [], []
        for chunk in chunks:
            self.calls += 1
            ents, rels = await self._fn(chunk)
            entities.extend(ents)
            relations.extend(rels)
        return entities, relations


class _FakeChunker:
    """Chunker that emits one Chunk per document using the document text."""

    async def chunk(self, docs):
        return [
            Chunk(
                text=d.text,
                document_id=d.id,
                document_source=d.source,
                chunk_index=0,
                chunk_size=len(d.text),
                cut_type="recursive",
            )
            for d in docs
        ]


def _record() -> GraphRecord:
    return GraphRecord(
        graph_id="med",
        neo4j=GraphConnection(uri="bolt://h", user_env="U", password_env="P"),
        vector=VectorConnection(url_env="URL"),
    )


def _target():
    """Return a fake IngestionTarget with mocked store/vector/embedder."""
    record = _record()
    embedder = SimpleNamespace(
        embed=AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts]),
        model_id="fake-model",
    )
    return SimpleNamespace(
        schema=record.graph_schema,
        collections=record.vector.collections,
        graph_store=SimpleNamespace(write_knowledge_graph=AsyncMock()),
        vector_store=SimpleNamespace(upsert=AsyncMock()),
        embedder=embedder,
    )


def _docs(n: int) -> list[Document]:
    return [
        Document(text=f"doc {i}", source="inline", external_id=str(i)) for i in range(n)
    ]


def _make_job(tmp_path, target, cache, sources, extractor):
    return IngestionJob(
        JobRecord(job_id=uuid4().hex, kind="ingestion", graph_id="med"),
        target=target,
        cache=cache,
        state_dir=tmp_path,
        sources=sources,
        extractor=extractor,
        chunker=_FakeChunker(),
    )


class TestIngestionRun:
    """Tests for run(), caching, and poison isolation."""

    async def test_run_marks_items_succeeded(self, tmp_path):
        """Verify a clean run marks every chunk succeeded."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        extractor = _FakeExtractor(fn)
        cache = IngestionCache(tmp_path / "med.db")
        source = _FakeCorpusSource(_docs(2))
        job = _make_job(tmp_path, _target(), cache, [source], extractor)
        record = await job.run()
        assert record.status == "succeeded"
        assert record.progress.succeeded == 2
        assert extractor.calls == 2

    async def test_extraction_cache_skips_extractor(self, tmp_path):
        """Verify a second job over the same content hits the extraction cache."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        extractor = _FakeExtractor(fn)
        cache = IngestionCache(tmp_path / "med.db")
        docs = _docs(2)
        await _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], extractor
        ).run()
        assert extractor.calls == 2
        await _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], extractor
        ).run()
        assert extractor.calls == 2

    async def test_embedding_cache_skips_embedder(self, tmp_path):
        """Verify a second job reuses cached embeddings (embedder not called)."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        cache = IngestionCache(tmp_path / "med.db")
        docs = _docs(2)
        target2 = _target()
        await _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], _FakeExtractor(fn)
        ).run()
        await _make_job(
            tmp_path, target2, cache, [_FakeCorpusSource(docs)], _FakeExtractor(fn)
        ).run()
        target2.embedder.embed.assert_not_called()

    async def test_poison_item_yields_partial(self, tmp_path):
        """Verify a chunk that always fails leaves the job partial with failures."""

        async def fn(chunk):
            if "poison" in chunk.text:
                raise RuntimeError("boom")
            return [Entity(name=chunk.text, label="Disease")], []

        cache = IngestionCache(tmp_path / "med.db")
        docs = [
            Document(text="good", source="inline", external_id="1"),
            Document(text="poison", source="inline", external_id="2"),
        ]
        job = _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(docs)], _FakeExtractor(fn)
        )
        record = await job.run()
        assert record.status == "partial"
        assert record.progress.failed == 1
        assert len(job.failures()) == 1

    async def test_resume_skips_succeeded(self, tmp_path):
        """Verify resuming a completed job skips already-succeeded chunks."""

        async def fn(chunk):
            return [Entity(name=chunk.text, label="Disease")], []

        extractor = _FakeExtractor(fn)
        cache = IngestionCache(tmp_path / "med.db")
        job = _make_job(
            tmp_path, _target(), cache, [_FakeCorpusSource(_docs(2))], extractor
        )
        await job.run()
        assert extractor.calls == 2
        record = await job.resume()
        assert record.status == "succeeded"
        assert record.progress.skipped == 2
        assert extractor.calls == 2
