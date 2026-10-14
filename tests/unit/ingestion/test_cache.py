"""Tests for the SQLite content-addressed ingestion cache.

Exercises key determinism, hit/miss round-trips for all three tables, the WAL
journal pragma, batched writes, and per-graph sharding/memoization. Uses a temp
directory; no external services.
"""

from __future__ import annotations

import sqlite3

import pytest

from am_diag.ingestion.cache import CacheManager, IngestionCache


class TestCacheKeys:
    """Tests for content-addressed key determinism."""

    def test_extraction_key_is_deterministic(self):
        """Verify the same inputs produce the same extraction key."""
        a = IngestionCache.extraction_key("text", "v1", "m1")
        b = IngestionCache.extraction_key("text", "v1", "m1")
        assert a == b

    def test_extraction_key_varies_with_schema(self):
        """Verify a different schema version yields a different key."""
        a = IngestionCache.extraction_key("text", "v1", "m1")
        b = IngestionCache.extraction_key("text", "v2", "m1")
        assert a != b

    def test_embedding_key_varies_with_model(self):
        """Verify a different embedding model yields a different key."""
        a = IngestionCache.embedding_key("text", "m1")
        b = IngestionCache.embedding_key("text", "m2")
        assert a != b


class TestCacheRoundTrips:
    """Tests for hit/miss round-trips across the three tables."""

    async def test_extraction_round_trip(self, tmp_path):
        """Verify an extraction payload writes and reads back."""
        cache = IngestionCache(tmp_path / "g.db")
        assert await cache.get_extraction("k") is None
        await cache.put_extraction_many([("k", {"entities": [1], "relations": []})])
        assert await cache.get_extraction("k") == {"entities": [1], "relations": []}

    async def test_embedding_round_trip(self, tmp_path):
        """Verify float32-packed vectors round-trip with batch reads."""
        cache = IngestionCache(tmp_path / "g.db")
        await cache.put_embeddings([("a", [0.5, -0.25]), ("b", [1.0, 2.0])])
        got = await cache.get_embeddings(["a", "b", "missing"])
        assert got["a"] == [0.5, -0.25]
        assert got["b"] == [1.0, 2.0]
        assert "missing" not in got

    async def test_answer_round_trip(self, tmp_path):
        """Verify an answer payload writes and reads back."""
        cache = IngestionCache(tmp_path / "g.db")
        await cache.put_answer("q", {"answer": "yes"})
        assert await cache.get_answer("q") == {"answer": "yes"}

    async def test_empty_batch_writes_are_noops(self, tmp_path):
        """Verify empty batch writes do not error."""
        cache = IngestionCache(tmp_path / "g.db")
        await cache.put_extraction_many([])
        await cache.put_embeddings([])
        assert await cache.get_embeddings([]) == {}


class TestCachePragmas:
    """Tests for the database-level WAL pragma."""

    async def test_wal_mode_enabled(self, tmp_path):
        """Verify the cache database is created in WAL journal mode on first use."""
        await IngestionCache(tmp_path / "g.db").get_answer("k")
        conn = sqlite3.connect(tmp_path / "g.db")
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        assert mode.lower() == "wal"


class TestCacheResilience:
    """Tests that cache contention degrades to a miss instead of an error."""

    async def test_locked_database_reads_miss_and_skips_writes(
        self, tmp_path, monkeypatch
    ):
        """Verify a write-locked database neither raises nor returns stale data."""
        monkeypatch.setattr("am_diag.ingestion.cache._BUSY_TIMEOUT_MS", 50)
        cache = IngestionCache(tmp_path / "g.db")
        await cache.put_answer("seed", {"answer": "kept"})
        blocker = sqlite3.connect(tmp_path / "g.db")
        try:
            blocker.execute("BEGIN IMMEDIATE")
            await cache.put_answer("q", {"answer": "lost"})
            await cache.put_embeddings([("e", [1.0])])
            assert await cache.get_answer("seed") == {"answer": "kept"}
        finally:
            blocker.rollback()
            blocker.close()
        assert await cache.get_answer("q") is None
        assert await cache.get_embeddings(["e"]) == {}

    async def test_embedding_lookup_handles_more_keys_than_sqlite_allows(
        self, tmp_path
    ):
        """Verify a lookup larger than one IN-clause batch returns every hit."""
        cache = IngestionCache(tmp_path / "g.db")
        keys = [f"k{i}" for i in range(2500)]
        await cache.put_embeddings([(key, [1.0]) for key in keys])
        assert len(await cache.get_embeddings(keys)) == 2500


class TestCacheManager:
    """Tests for per-graph sharding and memoization."""

    def test_for_graph_shards_per_graph_id(self, tmp_path):
        """Verify each graph gets its own database file."""
        manager = CacheManager(tmp_path)
        a = manager.for_graph("med")
        b = manager.for_graph("drug")
        assert a.db_path != b.db_path
        assert a.db_path.name == "med.db"

    def test_for_graph_memoizes(self, tmp_path):
        """Verify the same graph id returns the same cache instance."""
        manager = CacheManager(tmp_path)
        assert manager.for_graph("med") is manager.for_graph("med")

    @pytest.mark.parametrize("graph_id", ["../escape", "a/b", "", "..", ".hidden"])
    def test_for_graph_rejects_ids_that_leave_the_cache_dir(self, tmp_path, graph_id):
        """Verify graph ids with path components are refused."""
        with pytest.raises(ValueError, match="Invalid graph id"):
            CacheManager(tmp_path).for_graph(graph_id)
