"""SQLite content-addressed caches for ingestion and QA.

One database file per graph (``cache/{graph_id}.db``) holds three content-keyed
tables — extraction, embedding, and answer. The cache is an optimization, never
the source of truth: a miss or lock contention degrades to recompute, never to
wrong data (correctness is the idempotent MERGE/upsert plus durable job state).

Scaling rules enforced here:
  * WAL mode + ``busy_timeout`` so readers never block the single writer.
  * Every blocking ``sqlite3`` call, including first-use schema creation, runs in
    a thread executor (``asyncio.to_thread``) so the event loop is never stalled.
  * Lock or busy errors degrade to a miss (reads) or a skipped write.
  * Writes are batched: ``put_*_many`` commits one transaction per ingest batch.
  * Embeddings are float32-packed BLOBs (``dim * 4`` bytes), keeping the table
    compact at millions of rows.
"""

from __future__ import annotations

import array
import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 30_000
_MAX_QUERY_KEYS = 500
_GRAPH_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _sha256(*parts: str) -> str:
    """Return the hex sha256 of the parts joined by a null separator."""
    digest = hashlib.sha256()
    digest.update("\x00".join(parts).encode("utf-8"))
    return digest.hexdigest()


class IngestionCache:
    """Content-addressed SQLite cache for one graph.

    Args:
        db_path: Path to this graph's cache database file. Parent directories
            and the schema are created on first use, inside a worker thread.
    """

    def __init__(self, db_path: Path) -> None:
        """Remember the database path; the database opens on first use."""
        self._db_path = db_path
        self._init_lock = threading.Lock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        """Filesystem path of this cache's database file."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection with the per-connection pragmas applied."""
        self._ensure_initialized()
        return self._open()

    def _ensure_initialized(self) -> None:
        """Create the database file and tables once (called from worker threads)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._initialized = True

    def _open(self) -> sqlite3.Connection:
        """Open a raw connection with the busy timeout applied."""
        conn = sqlite3.connect(self._db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        return conn

    def _init_db(self) -> None:
        """Create the WAL journal and the three content-keyed tables."""
        conn = self._open()
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
    def answer_key(
        question: str,
        project_id: str,
        agent_config_hash: str,
        *,
        graphs: list[str] | None = None,
    ) -> str:
        """Return the content-addressed key for a QA answer.

        ``graphs`` is the selected graph subset; ``None`` means every graph and is
        kept distinct from any explicit subset.
        """
        graph_part = "*" if graphs is None else ",".join(sorted(graphs))
        return _sha256(question, project_id, agent_config_hash, graph_part)

    async def get_extraction(self, key: str) -> dict[str, Any] | None:
        """Return the cached extraction payload for ``key`` or ``None``."""
        return await asyncio.to_thread(self._get_extraction, key)

    def _get_extraction(self, key: str) -> dict[str, Any] | None:
        return self._get_payload("extraction", key)

    def _get_payload(self, table: str, key: str) -> dict[str, Any] | None:
        """Return the JSON payload stored under ``key``; ``None`` on miss or lock."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    f"SELECT payload FROM {table} WHERE key = ?",  # noqa: S608
                    (key,),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            logger.warning("Cache read from %s failed: %s", table, exc)
            return None
        return json.loads(row[0]) if row else None

    async def put_extraction_many(
        self, items: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Persist extraction payloads in a single transaction."""
        if not items:
            return
        await asyncio.to_thread(self._put_extraction_many, items)

    def _put_extraction_many(self, items: list[tuple[str, dict[str, Any]]]) -> None:
        self._write(
            "INSERT OR REPLACE INTO extraction (key, payload) VALUES (?, ?)",
            [(key, json.dumps(payload)) for key, payload in items],
        )

    def _write(self, statement: str, rows: list[tuple[Any, ...]]) -> None:
        """Run one write transaction; a lock or busy error skips the write."""
        try:
            conn = self._connect()
            try:
                conn.executemany(statement, rows)
                conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            logger.warning("Cache write skipped: %s", exc)

    async def get_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        """Return cached vectors for the subset of ``keys`` that are present."""
        if not keys:
            return {}
        return await asyncio.to_thread(self._get_embeddings, keys)

    def _get_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        rows: list[tuple[str, bytes]] = []
        try:
            conn = self._connect()
            try:
                for start in range(0, len(keys), _MAX_QUERY_KEYS):
                    chunk = keys[start : start + _MAX_QUERY_KEYS]
                    placeholders = ",".join("?" for _ in chunk)
                    rows.extend(
                        conn.execute(
                            "SELECT key, vector FROM embedding "  # noqa: S608
                            f"WHERE key IN ({placeholders})",
                            chunk,
                        ).fetchall()
                    )
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            logger.warning("Cache read from embedding failed: %s", exc)
            return {}
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
        self._write(
            "INSERT OR REPLACE INTO embedding (key, dim, vector) VALUES (?, ?, ?)",
            [(key, len(vec), array.array("f", vec).tobytes()) for key, vec in items],
        )

    async def get_answer(self, key: str) -> dict[str, Any] | None:
        """Return the cached answer payload for ``key`` or ``None``."""
        return await asyncio.to_thread(self._get_answer, key)

    def _get_answer(self, key: str) -> dict[str, Any] | None:
        return self._get_payload("answer", key)

    async def put_answer(self, key: str, answer: dict[str, Any]) -> None:
        """Persist one answer payload."""
        await asyncio.to_thread(self._put_answer, key, answer)

    def _put_answer(self, key: str, answer: dict[str, Any]) -> None:
        self._write(
            "INSERT OR REPLACE INTO answer (key, payload) VALUES (?, ?)",
            [(key, json.dumps(answer))],
        )


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
        """Return the memoized cache for ``graph_id`` (``cache/{graph_id}.db``).

        Raises:
            ValueError: If ``graph_id`` could point outside the cache directory.
        """
        if not _GRAPH_ID_RE.fullmatch(graph_id):
            raise ValueError(f"Invalid graph id for cache path: {graph_id!r}")
        cache = self._caches.get(graph_id)
        if cache is None:
            cache = IngestionCache(self._cache_dir / f"{graph_id}.db")
            self._caches[graph_id] = cache
        return cache

    async def close(self) -> None:
        """Drop all memoized caches (connections are per-call, nothing to close)."""
        self._caches.clear()
