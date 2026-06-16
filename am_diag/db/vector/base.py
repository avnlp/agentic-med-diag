"""Template-method base class for vector store clients.

``VectorStoreBase`` factors ALL shared logic (batching, filter normalization,
result mapping, lazy client initialization, async context manager) into the
base. Concrete subclasses implement only the ``_``-prefixed backend primitives.

Filter format accepted by ``search``, ``scroll``, and ``hybrid_search``:

    Filters are expressed as a flat ``dict[str, Any]`` where each key is a
    payload field name and each value is either:

    * a scalar (``str`` / ``int`` / ``bool``) — matched by exact equality, or
    * a ``list`` of scalars — matched by "any of" (OR within the field).

    All key-value pairs are combined with AND.  Example::

        {"kind": "disease", "source": ["pubmed", "pubmed_case_reports"]}
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import Self

from am_diag.common.data_models.enums import Distance
from am_diag.common.data_models.vector_record import VectorHit, VectorRecord


class VectorStoreBase(ABC):
    """Contract for vector store clients used in ingestion and query pipelines.

    Subclasses implement the ``_``-prefixed abstract primitives; all public
    interface methods are implemented here and delegate to those primitives.
    This ensures that batching, result mapping, and filter normalization logic
    live in exactly one place.
    """

    def __init__(self) -> None:
        """Initialize shared lazy-client state."""
        self._client: Any = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    async def client(self) -> Any:
        """Lazy-initialize the backend client with double-checked locking.

        Returns:
            The connected backend client instance.
        """
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = await self._connect()
        return self._client

    @abstractmethod
    async def _connect(self) -> Any:
        """Create and return the backend client connection.

        Called once on first access; the result is cached in ``self._client``.

        Returns:
            A connected backend client.
        """

    @abstractmethod
    async def _create_collection(
        self,
        name: str,
        vector_size: int,
        distance: Distance,
    ) -> None:
        """Backend-specific collection creation.

        Args:
            name: Collection name (backend-normalized if needed).
            vector_size: Dimensionality of stored vectors.
            distance: Distance metric for the collection index.
        """

    @abstractmethod
    async def _upsert_batch(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> None:
        """Write a single batch of records to the backend.

        Called once per batch slice from the public ``upsert`` method.

        Args:
            collection: Target collection name.
            records: Records in this batch slice.
        """

    @abstractmethod
    async def _search(
        self,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float | None,
        filters: Any,
    ) -> list[VectorHit]:
        """Execute a dense vector similarity search.

        Args:
            collection: Collection to search.
            vector: Query embedding.
            limit: Maximum results to return.
            score_threshold: Minimum score; lower-scoring results are dropped.
            filters: Backend-compiled filter from ``_compile_filter``.

        Returns:
            Hits ordered by descending score.
        """

    @abstractmethod
    async def _hybrid_search(
        self,
        collection: str,
        vector: list[float],
        text: str,
        limit: int,
        filters: Any,
        alpha: float,
    ) -> list[VectorHit]:
        """Execute a hybrid dense+sparse search.

        ``alpha`` controls the dense/sparse blend: 1.0 = pure dense,
        0.0 = pure sparse (BM25-style keyword).

        Args:
            collection: Collection to search.
            vector: Dense query embedding.
            text: Raw query text for the sparse leg.
            limit: Maximum results.
            filters: Backend-compiled filter.
            alpha: Dense weight in [0, 1].

        Returns:
            Hits ordered by descending fused score.
        """

    @abstractmethod
    async def _scroll(
        self,
        collection: str,
        limit: int,
        page_offset: str | None,
        filters: Any,
        with_vectors: bool,
    ) -> tuple[list[VectorRecord], str | None]:
        """Paginate through all records in a collection.

        Args:
            collection: Collection to paginate.
            limit: Page size.
            page_offset: Cursor from the previous call; ``None`` starts fresh.
            filters: Backend-compiled filter.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.

        Returns:
            ``(records, next_page_offset)``. ``next_page_offset`` is ``None``
            on the last page.
        """

    @abstractmethod
    async def _retrieve(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool,
    ) -> list[VectorRecord]:
        """Fetch specific records by ID.

        Args:
            collection: Collection that holds the records.
            ids: IDs to fetch.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.

        Returns:
            Found records in arbitrary order; missing IDs are silently omitted.
        """

    @abstractmethod
    async def _count(self, collection: str) -> int:
        """Return the total number of records in a collection.

        Args:
            collection: Collection name.

        Returns:
            Total record count.
        """

    @abstractmethod
    async def _delete(self, collection: str, ids: list[str]) -> None:
        """Remove records by ID.

        Args:
            collection: Collection that holds the records.
            ids: IDs to delete; non-existent IDs are silently ignored.
        """

    @abstractmethod
    async def _collection_exists(self, name: str) -> bool:
        """Check whether a collection exists in the backend.

        Args:
            name: Collection name.

        Returns:
            ``True`` if the collection exists, ``False`` otherwise.
        """

    @abstractmethod
    async def _delete_collection(self, name: str) -> None:
        """Permanently drop a collection and all its records.

        Args:
            name: Collection name to delete.
        """

    @abstractmethod
    def _compile_filter(self, filters: dict[str, Any]) -> Any:
        """Convert an am_diag filter dict to the backend's native filter type.

        Called by ``_build_filter`` only when ``filters`` is non-empty.

        Args:
            filters: Non-empty flat filter dict.

        Returns:
            Backend-native filter object.
        """

    @abstractmethod
    def _quantization_config(self, settings: Any) -> Any | None:
        """Build a backend-specific quantization config from settings.

        Args:
            settings: Backend-specific settings object.

        Returns:
            Quantization config, or ``None`` when quantization is disabled.
        """

    @property
    @abstractmethod
    def _distance_map(self) -> dict[Distance, Any]:
        """Map from ``Distance`` enum values to backend distance constants.

        Returns:
            Dict keyed by ``Distance`` enum, values are backend-specific.
        """

    def _build_filter(self, filters: dict[str, Any] | None) -> Any:
        """Normalize a possibly-None filter dict into a backend filter object.

        Handles the ``None`` / empty-dict cases here so every subclass
        ``_compile_filter`` can assume a non-empty dict.

        Args:
            filters: Filter dict from the caller, or ``None``.

        Returns:
            Backend-native filter, or ``None`` when no filter is needed.
        """
        if not filters:
            return None
        return self._compile_filter(filters)

    async def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create a new vector collection (index).

        Args:
            name: Collection name.
            vector_size: Dimensionality of stored vectors.
            distance: Distance metric. Defaults to ``Distance.COSINE``.
        """
        await self._create_collection(name, vector_size, distance)

    async def delete_collection(self, name: str) -> None:
        """Permanently drop a collection and all its records.

        Args:
            name: Collection name to delete.
        """
        await self._delete_collection(name)

    async def collection_exists(self, name: str) -> bool:
        """Check whether a collection exists.

        Args:
            name: Collection name.

        Returns:
            ``True`` if the collection exists, ``False`` otherwise.
        """
        return await self._collection_exists(name)

    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the collection only if it does not already exist.

        Args:
            name: Collection name.
            vector_size: Dimensionality of stored vectors.
            distance: Distance metric. Defaults to ``Distance.COSINE``.
        """
        if not await self.collection_exists(name):
            await self.create_collection(name, vector_size, distance)

    async def upsert(
        self,
        collection: str,
        records: list[VectorRecord],
        batch_size: int = 256,
    ) -> None:
        """Insert or update a batch of vector records.

        Existing records with the same ``id`` are overwritten. Splits
        ``records`` into slices of ``batch_size`` to avoid hitting backend
        request-size limits.

        Args:
            collection: Target collection name.
            records: Records to write.
            batch_size: Maximum records per individual backend call.
        """
        if not records:
            return
        for i in range(0, len(records), batch_size):
            await self._upsert_batch(collection, records[i : i + batch_size])

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Return the ``limit`` nearest neighbours to ``query_vector``.

        Args:
            collection: Collection to query.
            query_vector: Embedding to search with.
            limit: Maximum results to return.
            score_threshold: Minimum similarity score; lower-scoring results are
                dropped before returning.
            filters: Optional payload filter (see module docstring for format).

        Returns:
            Results ordered by descending similarity score.
        """
        return await self._search(
            collection,
            query_vector,
            limit,
            score_threshold,
            self._build_filter(filters),
        )

    async def hybrid_search(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        alpha: float = 0.5,
    ) -> list[VectorHit]:
        """Hybrid dense+sparse search.

        ``alpha`` controls the dense/sparse blend: 1.0 = pure dense,
        0.0 = pure sparse (BM25-style keyword).

        Args:
            collection: Collection to query.
            query_vector: Dense query embedding.
            query_text: Raw query text for the sparse search leg.
            limit: Maximum results to return.
            filters: Optional payload filter.
            alpha: Dense weight in [0, 1].

        Returns:
            Results ordered by descending fused score.
        """
        return await self._hybrid_search(
            collection,
            query_vector,
            query_text,
            limit,
            self._build_filter(filters),
            alpha,
        )

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        page_offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        """Paginate through all records in a collection.

        Args:
            collection: Collection to iterate over.
            limit: Maximum records per page.
            page_offset: Cursor returned by a previous call. Pass ``None``
                to start from the beginning.
            filters: Optional payload filter.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.

        Returns:
            A ``(records, next_page_offset)`` tuple. ``next_page_offset`` is
            ``None`` on the last page.
        """
        return await self._scroll(
            collection,
            limit,
            page_offset,
            self._build_filter(filters),
            with_vectors,
        )

    async def retrieve(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        """Fetch specific records by ID.

        Args:
            collection: Collection that holds the records.
            ids: IDs of records to fetch.
            with_vectors: When ``True``, populate ``VectorRecord.vector``.
                Defaults to ``False`` (vector is ``[]``).

        Returns:
            Found records in arbitrary order. Missing IDs are silently omitted.
        """
        if not ids:
            return []
        return await self._retrieve(collection, ids, with_vectors)

    async def count(self, collection: str) -> int:
        """Return the total number of records in a collection.

        Args:
            collection: Collection name.

        Returns:
            Total record count.
        """
        return await self._count(collection)

    async def delete(self, collection: str, ids: list[str]) -> None:
        """Remove records by ID.

        Args:
            collection: Collection that holds the records.
            ids: IDs of records to remove.
        """
        if not ids:
            return
        await self._delete(collection, ids)

    @abstractmethod
    async def close(self) -> None:
        """Release the underlying client connection and resources."""

    async def __aenter__(self) -> Self:
        """Enter async context; return self for use as an async context manager."""
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit async context; close the store and release resources."""
        await self.close()
