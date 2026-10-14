"""SQLite content-addressed caches for ingestion and QA.

One database file per graph (``cache/{graph_id}.db``) holds three content-keyed
tables — extraction, embedding, and answer. The cache is an optimization, never
the source of truth: a miss or lock contention degrades to recompute, never to
wrong data (correctness is the idempotent MERGE/upsert plus durable job state).

Scaling rules enforced here:
  * WAL mode + ``busy_timeout`` so readers never block the single writer.
  * Every blocking ``sqlite3`` call runs in a thread executor
    (``asyncio.to_thread``) so the event loop is never stalled.
  * Writes are batched: ``put_*_many`` commits one transaction per ingest batch.
  * Embeddings are float32-packed BLOBs (``dim * 4`` bytes), keeping the table
    compact at millions of rows.
"""

from __future__ import annotations

import array
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


_BUSY_TIMEOUT_MS = 30_000


def _sha256(*parts: str) -> str:
    """Return the hex sha256 of the parts joined by a null separator."""
    digest = hashlib.sha256()
    digest.update("\x00".join(parts).encode("utf-8"))
    return digest.hexdigest()


class IngestionCache:
    """Content-addressed SQLite cache for one graph.

    Args:
        db_path: Path to this graph's cache database file. Parent directories
            are created and the schema is initialized on construction.
    """

    def __init__(self, db_path: Path) -> None:
        """Open and initialize the cache database (WAL + tables)."""
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def db_path(self) -> Path:
        """Filesystem path of this cache's database file."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection with the per-connection pragmas applied."""
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        return conn

    def _init_db(self) -> None:
        """Create the WAL journal and the three content-keyed tables."""
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS extraction "
                "(key TEXT PRIMARY KEY, payload TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embedding "
                "(key TEXT PRIMARY KEY, dim INTEGER, vector BLOB)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS answer (key TEXT PRIMARY KEY, payload TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def extraction_key(content: str, schema_version: str, model_id: str) -> str:
        """Return the content-addressed key for an extraction result."""
        return _sha256(content, schema_version, model_id)

    @staticmethod
    def embedding_key(text: str, model_id: str) -> str:
        """Return the content-addressed key for an embedding."""
        return _sha256(text, model_id)

    @staticmethod
    def answer_key(question: str, project_id: str, agent_config_hash: str) -> str:
        """Return the content-addressed key for a QA answer."""
        return _sha256(question, project_id, agent_config_hash)

    async def get_extraction(self, key: str) -> dict[str, Any] | None:
        """Return the cached extraction payload for ``key`` or ``None``."""
        return await asyncio.to_thread(self._get_extraction, key)

    def _get_extraction(self, key: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload FROM extraction WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None

    async def put_extraction_many(
        self, items: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Persist extraction payloads in a single transaction."""
        if not items:
            return
        await asyncio.to_thread(self._put_extraction_many, items)

    def _put_extraction_many(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO extraction (key, payload) VALUES (?, ?)",
                [(key, json.dumps(payload)) for key, payload in items],
            )
            conn.commit()
        finally:
            conn.close()

    async def get_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        """Return cached vectors for the subset of ``keys`` that are present."""
        if not keys:
            return {}
        return await asyncio.to_thread(self._get_embeddings, keys)

    def _get_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in keys)
            rows = conn.execute(
                f"SELECT key, vector FROM embedding WHERE key IN ({placeholders})",
                keys,
            ).fetchall()
        finally:
            conn.close()
        result: dict[str, list[float]] = {}
        for key, blob in rows:
            vec = array.array("f")
            vec.frombytes(blob)
            result[key] = list(vec)
        return result

    async def put_embeddings(self, items: list[tuple[str, list[float]]]) -> None:
        """Persist float32-packed vectors in a single transaction."""
        if not items:
            return
        await asyncio.to_thread(self._put_embeddings, items)

    def _put_embeddings(self, items: list[tuple[str, list[float]]]) -> None:
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding (key, dim, vector) VALUES (?, ?, ?)",
                [
                    (key, len(vec), array.array("f", vec).tobytes())
                    for key, vec in items
                ],
            )
            conn.commit()
        finally:
            conn.close()

    async def get_answer(self, key: str) -> dict[str, Any] | None:
        """Return the cached answer payload for ``key`` or ``None``."""
        return await asyncio.to_thread(self._get_answer, key)

    def _get_answer(self, key: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload FROM answer WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None

    async def put_answer(self, key: str, answer: dict[str, Any]) -> None:
        """Persist one answer payload."""
        await asyncio.to_thread(self._put_answer, key, answer)

    def _put_answer(self, key: str, answer: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO answer (key, payload) VALUES (?, ?)",
                (key, json.dumps(answer)),
            )
            conn.commit()
        finally:
            conn.close()


class CacheManager:
    """Opens and memoizes one :class:`IngestionCache` per graph.

    Args:
        cache_dir: Directory under which per-graph database files live.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Store the cache root directory."""
        self._cache_dir = cache_dir
        self._caches: dict[str, IngestionCache] = {}

    def for_graph(self, graph_id: str) -> IngestionCache:
        """Return the memoized cache for ``graph_id`` (``cache/{graph_id}.db``)."""
        cache = self._caches.get(graph_id)
        if cache is None:
            cache = IngestionCache(self._cache_dir / f"{graph_id}.db")
            self._caches[graph_id] = cache
        return cache

    async def close(self) -> None:
        """Drop all memoized caches (connections are per-call, nothing to close)."""
        self._caches.clear()
