"""Concrete Qdrant vector store implementation.

Implements only the ``_``-prefixed backend primitives required by
``VectorStoreBase``. All shared logic (batching, filter normalization, result
mapping, lazy client init, async context manager) lives in the base class.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from qdrant_client.models import Distance as QdrantDistance

from am_diag.common.data_models.enums import Distance
from am_diag.common.data_models.vector_record import VectorHit, VectorRecord
from am_diag.db.vector.base import VectorStoreBase
from am_diag.db.vector.settings import QdrantSettings


logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStoreBase):
    """Async Qdrant client implementing ``VectorStoreBase``.

    Callers supply pre-computed vectors; no embedding is performed inside this
    component. Use ``create_qdrant_store`` as the preferred factory.

    Args:
        settings: Connection settings. Typically loaded from environment
            variables via ``QdrantSettings``.
    """

    def __init__(self, settings: QdrantSettings) -> None:
        """Initialise the Qdrant vector store client.

        The underlying ``AsyncQdrantClient`` is created lazily via the base
        class ``client`` property on first use.

        Args:
            settings: Connection settings from environment variables.
        """
        super().__init__()
        self._settings = settings

    @property
    def _distance_map(self) -> dict[Distance, QdrantDistance]:
        """Map from ``Distance`` enum values to Qdrant distance constants."""
        return {
            Distance.COSINE: QdrantDistance.COSINE,
            Distance.EUCLID: QdrantDistance.EUCLID,
            Distance.DOT: QdrantDistance.DOT,
        }

    async def _connect(self) -> AsyncQdrantClient:
        """Create and return a connected ``AsyncQdrantClient``.

        Returns:
            A ready-to-use ``AsyncQdrantClient``.
        """
        return AsyncQdrantClient(
            url=self._settings.url,
            api_key=self._settings.api_key or None,
        )

    # Retained for backward compatibility; tests inject via this name.
    async def _get_client(self) -> AsyncQdrantClient:
        """Lazy-initialise the Qdrant client with double-checked locking."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = await self._connect()
        return self._client  # type: ignore[return-value]

    def _quantization_config(self, settings: QdrantSettings) -> Any | None:
        """Build a Qdrant native quantization config from settings.

        Args:
            settings: Qdrant settings containing the quantization mode.

        Returns:
            A Qdrant quantization config object, or ``None``.
        """
        if settings.quantization == "scalar_int8":
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            )
        if settings.quantization == "binary":
            return BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))
        return None

    def _compile_filter(self, filters: dict[str, Any]) -> Filter:
        """Convert an am_diag filter dict into a Qdrant ``Filter`` object.

        Scalar values produce ``MatchValue`` (exact equality); list values
        produce ``MatchAny`` (OR within the field). All conditions are AND-ed
        via Qdrant's ``must`` list.

        Args:
            filters: Non-empty flat filter dict.

        Returns:
            A Qdrant ``Filter`` instance.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions)

    async def _create_collection(
        self,
        name: str,
        vector_size: int,
        distance: Distance,
    ) -> None:
        """Create a Qdrant collection with the given vector configuration.

        Args:
            name: Collection name.
            vector_size: Dimensionality of stored vectors.
            distance: Distance metric.
        """
        client = await self._get_client()
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self._distance_map[distance],
            ),
            quantization_config=self._quantization_config(self._settings),
        )
        logger.debug("Created Qdrant collection %r (size=%d)", name, vector_size)

    async def _upsert_batch(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> None:
        """Write one batch of records to a Qdrant collection.

        Args:
            collection: Target collection name.
            records: Records in this batch.
        """
        client = await self._get_client()
        qdrant_points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in records
        ]
        await client.upsert(collection_name=collection, points=qdrant_points)
        logger.debug("Upserted %d point(s) into %r", len(records), collection)

    async def _search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float | None,
        filters: Filter | None,
    ) -> list[VectorHit]:
        """Execute a dense vector similarity search via Qdrant's Query API.

        Args:
            collection: Collection to search.
            vector: Query embedding.
            limit: Maximum results.
            score_threshold: Minimum score.
            filters: Compiled Qdrant ``Filter``, or ``None``.

        Returns:
            Hits ordered by descending score.
        """
        client = await self._get_client()
        response = await client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filters,
            with_payload=True,
        )
        return [
            VectorHit(id=UUID(str(p.id)), score=p.score, payload=p.payload or {})
            for p in response.points
        ]

    async def _hybrid_search(
        self,
        collection: str,
        vector: list[float],
        text: str,
        limit: int,
        filters: Filter | None,
        alpha: float,
    ) -> list[VectorHit]:
        """Execute a hybrid dense+sparse search.

        Qdrant's native sparse-vector hybrid (Query API prefetch + fusion)
        requires a named sparse vector configured at collection creation time.
        Until the sparse vector infrastructure is provisioned in this project,
        this method falls back to the dense search leg so callers can switch to
        true hybrid simply by enabling sparse vectors without changing call
        sites.

        TODO: Wire Qdrant sparse vector (BM42 or custom) once collection schema
        includes a named sparse field.

        Args:
            collection: Collection to search.
            vector: Dense query embedding.
            text: Raw query text (unused in fallback path).
            limit: Maximum results.
            filters: Compiled Qdrant ``Filter``, or ``None``.
            alpha: Dense weight (ignored in fallback; dense weight = 1.0).

        Returns:
            Hits ordered by descending score.
        """
        return await self._search(collection, vector, limit, None, filters)

    async def _scroll(
        self,
        collection: str,
        limit: int,
        page_offset: str | None,
        filters: Filter | None,
        with_vectors: bool,
    ) -> tuple[list[VectorRecord], str | None]:
        """Paginate through all records in a Qdrant collection.

        Args:
            collection: Collection to paginate.
            limit: Page size.
            page_offset: Cursor from the previous call; ``None`` starts fresh.
            filters: Compiled Qdrant ``Filter``, or ``None``.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.

        Returns:
            ``(records, next_page_offset)`` where ``next_page_offset`` is
            ``None`` on the last page.
        """
        client = await self._get_client()
        records, next_offset = await client.scroll(
            collection_name=collection,
            scroll_filter=filters,
            limit=limit,
            offset=page_offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        points = [
            VectorRecord(
                id=UUID(str(rec.id)),
                # rec.vector is list[float] when returned; cast silences variance check.
                vector=cast(list[float], rec.vector)
                if with_vectors and rec.vector
                else [],
                payload=rec.payload or {},
            )
            for rec in records
        ]
        return points, str(next_offset) if next_offset is not None else None

    async def _retrieve(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool,
    ) -> list[VectorRecord]:
        """Fetch specific records by ID from a Qdrant collection.

        Args:
            collection: Collection that holds the records.
            ids: IDs to fetch.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.

        Returns:
            Found records in arbitrary order.
        """
        client = await self._get_client()
        records = await client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            VectorRecord(
                id=UUID(str(rec.id)),
                vector=cast(list[float], rec.vector)
                if with_vectors and rec.vector
                else [],
                payload=rec.payload or {},
            )
            for rec in records
        ]

    async def _count(self, collection: str) -> int:
        """Return the exact total point count in a Qdrant collection.

        Args:
            collection: Collection name.

        Returns:
            Total point count.
        """
        client = await self._get_client()
        result = await client.count(collection_name=collection, exact=True)
        return result.count

    async def _delete(self, collection: str, ids: list[str]) -> None:
        """Delete points by ID from a Qdrant collection.

        Args:
            collection: Collection that holds the points.
            ids: IDs to delete; non-existent IDs are silently ignored.
        """
        client = await self._get_client()
        await client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=list(ids)),  # type: ignore[arg-type]
        )
        logger.debug("Deleted %d point(s) from %r", len(ids), collection)

    async def _collection_exists(self, name: str) -> bool:
        """Return ``True`` if the named collection exists in Qdrant.

        Args:
            name: Collection name.

        Returns:
            ``True`` if the collection exists.
        """
        client = await self._get_client()
        return await client.collection_exists(collection_name=name)

    async def _delete_collection(self, name: str) -> None:
        """Permanently drop a collection from Qdrant.

        Args:
            name: Collection name to delete.
        """
        client = await self._get_client()
        await client.delete_collection(collection_name=name)
        logger.debug("Deleted Qdrant collection %r", name)

    async def close(self) -> None:
        """Close the underlying Qdrant HTTP connection and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Module-level helpers retained for backward compatibility with existing tests
# that import them directly.


def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Convert a flat payload-filter dict into a Qdrant ``Filter`` object.

    This top-level helper is a convenience wrapper around
    ``QdrantVectorStore._compile_filter`` for callers that don't have a store
    instance but need the filter object (e.g. unit tests).

    Args:
        filters: Flat filter dict, or ``None`` for no filter.

    Returns:
        A Qdrant ``Filter`` object, or ``None`` when no filter is applied.
    """
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions)


def _build_quantization_config(settings: QdrantSettings) -> Any | None:
    """Build Qdrant native quantization config from settings.

    Retained for backward compatibility with existing tests.

    Args:
        settings: Qdrant settings.

    Returns:
        Quantization config, or ``None``.
    """
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._settings = settings
    return store._quantization_config(settings)


def create_qdrant_store(settings: QdrantSettings | None = None) -> QdrantVectorStore:
    """Instantiate a ``QdrantVectorStore`` from settings.

    Args:
        settings: Explicit settings object. When ``None``, settings are read
            from the environment (``QDRANT_URL``, ``QDRANT_API_KEY``).

    Returns:
        A configured ``QdrantVectorStore`` ready for use.
    """
    return QdrantVectorStore(settings or QdrantSettings())
