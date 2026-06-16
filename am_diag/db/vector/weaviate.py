"""Concrete Weaviate vector store implementation.

Implements only the ``_``-prefixed backend primitives required by
``VectorStoreBase``. All shared logic (batching, filter normalization, result
mapping, lazy client init, async context manager) lives in the base class.

Weaviate is the primary hybrid-search backend: native ``coll.query.hybrid``
is used for ``_hybrid_search`` without any fallback.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, cast
from urllib.parse import urlparse


# Optional dependency: weaviate-client
try:
    import weaviate
    import weaviate.classes as wvc
except ImportError as _exc:
    raise ImportError(
        "Weaviate support requires: pip install 'am_diag[weaviate]'",
    ) from _exc

from am_diag.common.data_models.enums import Distance
from am_diag.common.data_models.vector_record import VectorHit, VectorRecord
from am_diag.db.vector.base import VectorStoreBase
from am_diag.db.vector.settings import WeaviateSettings


logger = logging.getLogger(__name__)

# Fixed namespace for deterministic UUID5 derivation from caller string IDs.
# Never change this value — doing so would invalidate all stored object IDs.
_WEAVIATE_NS = uuid.UUID("7bdc6d62-da71-4dde-8ef5-8a5e3c7c4e2a")


def _to_collection_name(name: str) -> str:
    """Convert snake_case or kebab-case to Weaviate-required PascalCase.

    Weaviate raises ``WeaviateInvalidInputError`` for names starting with a
    lowercase letter. Centralising the conversion keeps all call sites clean.

    Args:
        name: Caller-supplied collection name in any casing.

    Returns:
        PascalCase collection name safe to pass to the Weaviate API.
    """
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def _to_weaviate_uuid(caller_id: str) -> str:
    """Derive a stable UUID5 from a caller string ID.

    Args:
        caller_id: Any string identifier supplied by the caller.

    Returns:
        A UUID5 string derived from ``_WEAVIATE_NS`` and ``caller_id``.
    """
    return str(uuid.uuid5(_WEAVIATE_NS, caller_id))


async def _connect(settings: WeaviateSettings) -> weaviate.WeaviateAsyncClient:
    """Build and connect a ``WeaviateAsyncClient`` based on the settings URL.

    Three deployment types are auto-detected from ``settings.url``:

    * ``localhost`` / ``127.0.0.1`` → ``weaviate.use_async_with_local``
    * ``*.weaviate.network`` / ``*.weaviate.cloud``
        → ``weaviate.use_async_with_weaviate_cloud``
    * anything else → ``weaviate.use_async_with_custom``

    Args:
        settings: Weaviate connection settings.

    Returns:
        A connected ``WeaviateAsyncClient``.
    """
    url = settings.url.rstrip("/")
    grpc_port = settings.grpc_port or None  # 0 → None disables gRPC

    if "localhost" in url or "127.0.0.1" in url:
        parsed = urlparse(url)
        http_port = parsed.port or 8080
        client = weaviate.use_async_with_local(port=http_port, grpc_port=grpc_port or 0)
    elif ".weaviate.network" in url or ".weaviate.cloud" in url:
        client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=url,
            auth_credentials=wvc.init.Auth.api_key(settings.api_key),
        )
    else:
        parsed = urlparse(url)
        is_secure = parsed.scheme == "https"
        client = weaviate.use_async_with_custom(
            http_host=parsed.hostname or "localhost",
            http_port=parsed.port or 8080,
            http_secure=is_secure,
            grpc_host=parsed.hostname or "localhost",
            grpc_port=grpc_port or 50051,
            grpc_secure=is_secure,
            auth_credentials=(
                wvc.init.Auth.api_key(settings.api_key) if settings.api_key else None
            ),
        )

    await client.connect()
    return client


def _build_quantizer_config(settings: WeaviateSettings) -> Any | None:
    """Build a Weaviate native vector-index quantizer config from settings.

    Args:
        settings: Weaviate settings containing the quantization mode.

    Returns:
        A Weaviate quantizer config, or ``None``.
    """
    quantizer = wvc.config.Configure.VectorIndex.Quantizer
    if settings.quantization == "sq":
        return quantizer.sq()
    if settings.quantization == "bq":
        return quantizer.bq()
    if settings.quantization == "rq":
        return quantizer.rq()
    if settings.quantization == "pq":
        return quantizer.pq()
    return None


def _build_filter(filters: dict[str, Any] | None) -> Any | None:
    """Convert the base-class filter dict to a Weaviate ``Filter`` object.

    Retained as a module-level function for backward compatibility with
    existing tests that import it directly.

    Args:
        filters: Flat filter dict, or ``None`` for no filter.

    Returns:
        A Weaviate ``Filter`` object, or ``None`` when no filter is applied.
    """
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if isinstance(value, list):
            conditions.append(wvc.query.Filter.by_property(key).contains_any(value))
        else:
            conditions.append(wvc.query.Filter.by_property(key).equal(value))
    result = conditions[0]
    for cond in conditions[1:]:
        result = result & cond
    return result


class WeaviateVectorStore(VectorStoreBase):
    """Async Weaviate client implementing ``VectorStoreBase``.

    Callers supply pre-computed vectors; no embedding is performed inside this
    component. String IDs are mapped to deterministic UUID5 values internally
    so Weaviate's UUID requirement is transparent to callers.

    Weaviate is the primary hybrid-search backend: ``_hybrid_search`` uses
    native ``coll.query.hybrid`` without fallback.

    Use ``create_weaviate_store`` as the preferred factory.

    Args:
        settings: Connection settings. Typically loaded from environment
            variables via ``WeaviateSettings``.
    """

    def __init__(self, settings: WeaviateSettings) -> None:
        """Initialise the Weaviate vector store client.

        The underlying ``WeaviateAsyncClient`` is connected lazily via the base
        class ``client`` property on first use.

        Args:
            settings: Connection settings from environment variables.
        """
        super().__init__()
        self._settings = settings

    @property
    def _distance_map(self) -> dict[Distance, Any]:
        """Map from ``Distance`` enum values to Weaviate distance constants."""
        return {
            Distance.COSINE: wvc.config.VectorDistances.COSINE,
            Distance.EUCLID: wvc.config.VectorDistances.L2_SQUARED,
            Distance.DOT: wvc.config.VectorDistances.DOT,
        }

    async def _connect(self) -> weaviate.WeaviateAsyncClient:
        """Create and return a connected ``WeaviateAsyncClient``.

        Returns:
            A connected ``WeaviateAsyncClient``.
        """
        return await _connect(self._settings)

    # Backward-compatible name used by existing tests.
    async def _get_client(self) -> weaviate.WeaviateAsyncClient:
        """Lazy-connect with double-checked locking."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = await self._connect()
        return self._client  # type: ignore[return-value]

    def _quantization_config(self, settings: WeaviateSettings) -> Any | None:
        """Build a Weaviate quantizer config from settings.

        Args:
            settings: Weaviate settings.

        Returns:
            Quantizer config, or ``None``.
        """
        return _build_quantizer_config(settings)

    def _compile_filter(self, filters: dict[str, Any]) -> Any:
        """Convert an am_diag filter dict to a Weaviate ``Filter`` object.

        Scalar values produce ``equal`` conditions; list values produce
        ``contains_any`` conditions. All conditions are AND-ed via ``&``.

        Args:
            filters: Non-empty flat filter dict.

        Returns:
            A Weaviate ``Filter`` instance.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(wvc.query.Filter.by_property(key).contains_any(value))
            else:
                conditions.append(wvc.query.Filter.by_property(key).equal(value))
        result = conditions[0]
        for cond in conditions[1:]:
            result = result & cond
        return result

    async def _create_collection(
        self,
        name: str,
        vector_size: int,
        distance: Distance,
    ) -> None:
        """Create a Weaviate collection; idempotent (no-op if already exists).

        ``vector_size`` is informational only — Weaviate infers the actual
        dimensionality from the first inserted vector.

        Args:
            name: Collection name (any casing; converted to PascalCase).
            vector_size: Dimensionality of stored vectors (informational).
            distance: Distance metric for the HNSW index.
        """
        client = await self._get_client()
        coll_name = _to_collection_name(name)
        if not await client.collections.exists(coll_name):
            await client.collections.create(
                name=coll_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=self._distance_map[distance],
                    quantizer=_build_quantizer_config(self._settings),
                ),
                properties=[
                    wvc.config.Property(
                        name="original_id",
                        data_type=wvc.config.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                    wvc.config.Property(
                        name="payload",
                        data_type=wvc.config.DataType.TEXT,
                        skip_vectorization=True,
                    ),
                ],
            )
            logger.debug(
                "Created Weaviate collection %r (distance=%s)",
                coll_name,
                distance,
            )

    async def _upsert_batch(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> None:
        """Write one batch via delete-then-insert to achieve upsert semantics.

        Weaviate has no native upsert-by-id; deleting before inserting is the
        recommended pattern and keeps the operation idempotent.

        Args:
            collection: Target collection name (any casing).
            records: Records in this batch.
        """
        client = await self._get_client()
        coll = client.collections.get(_to_collection_name(collection))
        uuids = [_to_weaviate_uuid(str(p.id)) for p in records]
        await coll.data.delete_many(
            where=wvc.query.Filter.by_id().contains_any(uuids),
        )
        objects = [
            wvc.data.DataObject(
                uuid=_to_weaviate_uuid(str(p.id)),
                vector=p.vector,
                properties={
                    "original_id": str(p.id),
                    "payload": json.dumps(p.payload),
                },
            )
            for p in records
        ]
        await coll.data.insert_many(objects)
        logger.debug("Upserted %d object(s) into %r", len(records), collection)

    async def _search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float | None,
        filters: Any,
    ) -> list[VectorHit]:
        """Execute a dense near-vector search.

        Weaviate returns distances, not similarities; scores are converted
        via ``1 - distance`` so that higher score means more similar,
        consistent with the base class contract.

        Args:
            collection: Collection to search.
            vector: Query embedding.
            limit: Maximum results.
            score_threshold: Minimum score after conversion.
            filters: Compiled Weaviate ``Filter``, or ``None``.

        Returns:
            Hits ordered by descending similarity score.
        """
        client = await self._get_client()
        coll = client.collections.get(_to_collection_name(collection))
        response = await coll.query.near_vector(
            near_vector=vector,
            limit=limit,
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(distance=True),
            return_properties=["original_id", "payload"],
        )
        results = [
            VectorHit(
                id=uuid.UUID(cast(str, obj.properties["original_id"])),
                score=1.0
                - (obj.metadata.distance if obj.metadata.distance is not None else 0.0),
                payload=json.loads(cast(str, obj.properties["payload"])),
            )
            for obj in response.objects
        ]
        if score_threshold is not None:
            results = [r for r in results if r.score >= score_threshold]
        return results

    async def _hybrid_search(
        self,
        collection: str,
        vector: list[float],
        text: str,
        limit: int,
        filters: Any,
        alpha: float,
    ) -> list[VectorHit]:
        """Execute a native Weaviate hybrid dense+BM25 search.

        Weaviate's ``coll.query.hybrid`` fuses dense and sparse scores
        natively. ``alpha=1.0`` is pure dense; ``alpha=0.0`` is pure BM25.

        Args:
            collection: Collection to search.
            vector: Dense query embedding.
            text: Raw query text for the BM25 leg.
            limit: Maximum results.
            filters: Compiled Weaviate ``Filter``, or ``None``.
            alpha: Dense weight in [0, 1].

        Returns:
            Hits ordered by descending fused score.
        """
        client = await self._get_client()
        coll = client.collections.get(_to_collection_name(collection))
        response = await coll.query.hybrid(
            query=text,
            vector=vector,
            alpha=alpha,
            limit=limit,
            filters=filters,
            return_properties=["original_id", "payload"],
        )
        return [
            VectorHit(
                id=uuid.UUID(cast(str, obj.properties["original_id"])),
                # Weaviate hybrid returns a score directly (already fused).
                score=(
                    obj.metadata.score if obj.metadata and obj.metadata.score else 0.0
                ),
                payload=json.loads(cast(str, obj.properties["payload"])),
            )
            for obj in response.objects
        ]

    async def _scroll(
        self,
        collection: str,
        limit: int,
        page_offset: str | None,
        filters: Any,
        with_vectors: bool,
    ) -> tuple[list[VectorRecord], str | None]:
        """Paginate through all objects using Weaviate's cursor-based pagination.

        ``page_offset`` is the UUID string of the last object from the previous
        page; pass ``None`` to start from the beginning.

        Args:
            collection: Collection to iterate over.
            limit: Maximum objects per page.
            page_offset: Cursor from a previous call.
            filters: Compiled Weaviate ``Filter``, or ``None``.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.

        Returns:
            ``(records, next_page_offset)``. ``next_page_offset`` is ``None``
            when the last page has been reached.
        """
        client = await self._get_client()
        coll = client.collections.get(_to_collection_name(collection))
        response = await coll.query.fetch_objects(
            limit=limit,
            after=page_offset,
            filters=filters,
            include_vector=with_vectors,
            return_properties=["original_id", "payload"],
        )
        points = [
            VectorRecord(
                id=uuid.UUID(cast(str, obj.properties["original_id"])),
                vector=(
                    list(cast(dict, obj.vector).get("default", []))
                    if with_vectors and obj.vector
                    else []
                ),
                payload=json.loads(cast(str, obj.properties["payload"])),
            )
            for obj in response.objects
        ]
        # When a full page is returned there may be more; use last UUID as cursor.
        next_offset = (
            str(response.objects[-1].uuid) if len(response.objects) == limit else None
        )
        return points, next_offset

    async def _retrieve(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool,
    ) -> list[VectorRecord]:
        """Fetch specific objects by their original string IDs.

        Args:
            collection: Collection that holds the objects.
            ids: Caller-supplied string IDs to fetch.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.

        Returns:
            Objects in arbitrary order. Missing IDs are silently omitted.
        """
        client = await self._get_client()
        coll = client.collections.get(_to_collection_name(collection))
        uuids = [_to_weaviate_uuid(id_) for id_ in ids]
        response = await coll.query.fetch_objects(
            filters=wvc.query.Filter.by_id().contains_any(uuids),
            include_vector=with_vectors,
            return_properties=["original_id", "payload"],
        )
        return [
            VectorRecord(
                id=uuid.UUID(cast(str, obj.properties["original_id"])),
                vector=(
                    list(cast(dict, obj.vector).get("default", []))
                    if with_vectors and obj.vector
                    else []
                ),
                payload=json.loads(cast(str, obj.properties["payload"])),
            )
            for obj in response.objects
        ]

    async def _count(self, collection: str) -> int:
        """Return the total number of objects in a collection.

        Args:
            collection: Collection name (any casing).

        Returns:
            Total object count.
        """
        client = await self._get_client()
        coll = client.collections.get(_to_collection_name(collection))
        result = await coll.aggregate.over_all(total_count=True)
        return result.total_count or 0

    async def _delete(self, collection: str, ids: list[str]) -> None:
        """Remove objects by their original string IDs.

        Args:
            collection: Collection that holds the objects.
            ids: Caller-supplied string IDs to delete.
        """
        client = await self._get_client()
        coll = client.collections.get(_to_collection_name(collection))
        weaviate_uuids = [_to_weaviate_uuid(id_) for id_ in ids]
        await coll.data.delete_many(
            where=wvc.query.Filter.by_id().contains_any(weaviate_uuids),
        )
        logger.debug("Deleted %d object(s) from %r", len(ids), collection)

    async def _collection_exists(self, name: str) -> bool:
        """Return ``True`` if the named collection exists.

        Args:
            name: Collection name (any casing).

        Returns:
            ``True`` if the collection exists.
        """
        client = await self._get_client()
        return await client.collections.exists(_to_collection_name(name))

    async def _delete_collection(self, name: str) -> None:
        """Permanently drop a collection and all its objects.

        Non-existent collections are silently ignored.

        Args:
            name: Collection name (any casing).
        """
        client = await self._get_client()
        coll_name = _to_collection_name(name)
        if await client.collections.exists(coll_name):
            await client.collections.delete(coll_name)
            logger.debug("Deleted Weaviate collection %r", coll_name)

    async def close(self) -> None:
        """Disconnect from Weaviate and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None


def create_weaviate_store(
    settings: WeaviateSettings | None = None,
) -> WeaviateVectorStore:
    """Instantiate a ``WeaviateVectorStore`` from settings.

    Args:
        settings: Explicit settings object. When ``None``, settings are read
            from the environment (``WEAVIATE_URL``, ``WEAVIATE_API_KEY``,
            ``WEAVIATE_GRPC_PORT``, ``WEAVIATE_BATCH_SIZE``).

    Returns:
        A configured ``WeaviateVectorStore`` ready for use.
    """
    return WeaviateVectorStore(settings or WeaviateSettings())
