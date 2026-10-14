"""Unit tests for QdrantVectorStore with mocked AsyncQdrantClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from am_diag.common.data_models.enums import Distance
from am_diag.common.data_models.vector_record import VectorHit, VectorRecord
from am_diag.db.vector.qdrant import (
    QdrantVectorStore,
    _build_filter,
    _build_quantization_config,
    create_qdrant_store,
)
from am_diag.db.vector.settings import QdrantSettings


_SETTINGS = QdrantSettings(url="http://localhost:6333", api_key="")

_E1 = uuid4()
_E2 = uuid4()


def _make_store(
    settings: QdrantSettings | None = None,
) -> tuple[QdrantVectorStore, MagicMock]:
    """Return a store with _client pre-set to a mock (bypasses lazy init)."""
    mock_client = MagicMock()
    mock_client.create_collection = AsyncMock()
    mock_client.delete_collection = AsyncMock()
    mock_client.collection_exists = AsyncMock(return_value=False)
    mock_client.upsert = AsyncMock()
    mock_client.retrieve = AsyncMock(return_value=[])
    mock_client.count = AsyncMock()
    mock_client.query_points = AsyncMock()
    mock_client.scroll = AsyncMock(return_value=([], None))
    mock_client.delete = AsyncMock()
    mock_client.close = AsyncMock()

    store = QdrantVectorStore(settings or _SETTINGS)
    store._client = mock_client  # inject: bypass lazy _get_client()
    return store, mock_client


def _query_response(points: list) -> MagicMock:
    """Build a mock ``QueryResponse`` with the given ``points`` list."""
    resp = MagicMock()
    resp.points = points
    return resp


class TestBuildFilter:
    """Tests for _build_filter helper."""

    def test_none_returns_none(self):
        """Verify None input returns None."""
        assert _build_filter(None) is None

    def test_empty_dict_returns_none(self):
        """Verify empty dict returns None."""
        assert _build_filter({}) is None

    def test_scalar_value_produces_match_value(self):
        """Verify scalar filter values produce MatchValue conditions."""
        from qdrant_client.models import MatchValue

        f = _build_filter({"kind": "disease"})
        assert f is not None
        assert f.must is not None
        assert len(f.must) == 1
        cond = f.must[0]
        assert cond.key == "kind"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "disease"

    def test_list_value_produces_match_any(self):
        """Verify list filter values produce MatchAny conditions."""
        from qdrant_client.models import MatchAny

        f = _build_filter({"source": ["pubmed", "pubmed_case_reports"]})
        assert f is not None
        assert f.must is not None
        cond = f.must[0]
        assert isinstance(cond.match, MatchAny)
        assert cond.match.any == ["pubmed", "pubmed_case_reports"]

    def test_multiple_keys_all_in_must(self):
        """Verify multiple filter keys all appear in the must list."""
        f = _build_filter({"label": "entity", "source": "pubmed"})
        assert f is not None
        assert len(f.must) == 2

    def test_bool_value_is_scalar(self):
        """Verify boolean filter values produce MatchValue conditions."""
        from qdrant_client.models import MatchValue

        f = _build_filter({"verified": True})
        assert f.must is not None
        assert isinstance(f.must[0].match, MatchValue)


class TestQdrantVectorStoreLazyInit:
    """Tests for QdrantVectorStore lazy initialisation."""

    def test_client_is_none_before_first_call(self):
        """Verify _client is None before _get_client is called."""
        store = QdrantVectorStore(_SETTINGS)
        assert store._client is None

    async def test_get_client_creates_client(self):
        """Verify _get_client creates a new client on first call."""
        with patch("am_diag.db.vector.qdrant.AsyncQdrantClient") as mock_cls:
            store = QdrantVectorStore(_SETTINGS)
            client = await store._get_client()
            mock_cls.assert_called_once_with(url=_SETTINGS.url, api_key=None)
            assert client is mock_cls.return_value

    async def test_get_client_reuses_existing(self):
        """Verify _get_client reuses an existing client on subsequent calls."""
        with patch("am_diag.db.vector.qdrant.AsyncQdrantClient") as mock_cls:
            store = QdrantVectorStore(_SETTINGS)
            c1 = await store._get_client()
            c2 = await store._get_client()
            mock_cls.assert_called_once()
            assert c1 is c2


class TestQdrantVectorStoreCreateCollection:
    """Tests for QdrantVectorStore.create_collection."""

    async def test_delegates_to_client(self):
        """Verify create_collection delegates to the Qdrant client."""
        store, mock_client = _make_store()
        await store.create_collection("entities", vector_size=1536)
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "entities"

    async def test_distance_mapping_cosine(self):
        """Verify COSINE distance is mapped correctly."""
        from qdrant_client.models import Distance as QdrantDistance

        store, mock_client = _make_store()
        await store.create_collection("col", 768, Distance.COSINE)
        params = mock_client.create_collection.call_args.kwargs["vectors_config"]
        assert params.distance == QdrantDistance.COSINE

    async def test_distance_mapping_euclid(self):
        """Verify EUCLID distance is mapped correctly."""
        from qdrant_client.models import Distance as QdrantDistance

        store, mock_client = _make_store()
        await store.create_collection("col", 768, Distance.EUCLID)
        params = mock_client.create_collection.call_args.kwargs["vectors_config"]
        assert params.distance == QdrantDistance.EUCLID

    async def test_distance_mapping_dot(self):
        """Verify DOT distance is mapped correctly."""
        from qdrant_client.models import Distance as QdrantDistance

        store, mock_client = _make_store()
        await store.create_collection("col", 768, Distance.DOT)
        params = mock_client.create_collection.call_args.kwargs["vectors_config"]
        assert params.distance == QdrantDistance.DOT

    async def test_no_quantization_by_default(self):
        """Verify quantization config is None by default."""
        store, mock_client = _make_store()
        await store.create_collection("col", 768)
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["quantization_config"] is None

    async def test_scalar_int8_quantization_passed_through(self):
        """Verify scalar INT8 quantization config is passed to the client."""
        from qdrant_client.models import ScalarQuantization, ScalarType

        settings = QdrantSettings(
            url="http://localhost:6333",
            api_key="",
            quantization="scalar_int8",
        )
        store, mock_client = _make_store(settings)
        await store.create_collection("col", 768)
        config = mock_client.create_collection.call_args.kwargs["quantization_config"]
        assert isinstance(config, ScalarQuantization)
        assert config.scalar.type == ScalarType.INT8

    async def test_binary_quantization_passed_through(self):
        """Verify binary quantization config is passed to the client."""
        from qdrant_client.models import BinaryQuantization

        settings = QdrantSettings(
            url="http://localhost:6333",
            api_key="",
            quantization="binary",
        )
        store, mock_client = _make_store(settings)
        await store.create_collection("col", 768)
        config = mock_client.create_collection.call_args.kwargs["quantization_config"]
        assert isinstance(config, BinaryQuantization)


class TestBuildQuantizationConfig:
    """Tests for _build_quantization_config helper."""

    def test_none_returns_none(self):
        """Verify 'none' quantization returns None."""
        settings = QdrantSettings(
            url="http://localhost:6333",
            api_key="",
            quantization="none",
        )
        assert _build_quantization_config(settings) is None

    def test_scalar_int8_builds_scalar_quantization(self):
        """Verify 'scalar_int8' builds ScalarQuantization with INT8."""
        from qdrant_client.models import ScalarQuantization, ScalarType

        settings = QdrantSettings(
            url="http://localhost:6333",
            api_key="",
            quantization="scalar_int8",
        )
        config = _build_quantization_config(settings)
        assert isinstance(config, ScalarQuantization)
        assert config.scalar.type == ScalarType.INT8
        assert config.scalar.always_ram is True

    def test_binary_builds_binary_quantization(self):
        """Verify 'binary' builds BinaryQuantization."""
        from qdrant_client.models import BinaryQuantization

        settings = QdrantSettings(
            url="http://localhost:6333",
            api_key="",
            quantization="binary",
        )
        config = _build_quantization_config(settings)
        assert isinstance(config, BinaryQuantization)
        assert config.binary.always_ram is True


class TestQdrantVectorStoreDeleteCollection:
    """Tests for QdrantVectorStore.delete_collection."""

    async def test_delegates_to_client(self):
        """Verify delete_collection delegates to the Qdrant client."""
        store, mock_client = _make_store()
        await store.delete_collection("old_collection")
        mock_client.delete_collection.assert_called_once_with(
            collection_name="old_collection",
        )


class TestQdrantVectorStoreCollectionExists:
    """Tests for QdrantVectorStore.collection_exists."""

    async def test_returns_false_when_missing(self):
        """Verify returns False for missing collections."""
        store, mock_client = _make_store()
        mock_client.collection_exists.return_value = False
        assert await store.collection_exists("missing") is False

    async def test_returns_true_when_present(self):
        """Verify returns True for existing collections."""
        store, mock_client = _make_store()
        mock_client.collection_exists.return_value = True
        assert await store.collection_exists("present") is True


class TestQdrantVectorStoreEnsureCollection:
    """Tests for QdrantVectorStore.ensure_collection."""

    async def test_creates_when_absent(self):
        """Verify ensure_collection creates collection when absent."""
        store, mock_client = _make_store()
        mock_client.collection_exists.return_value = False
        await store.ensure_collection("new_col", 1536)
        mock_client.create_collection.assert_called_once()

    async def test_skips_when_present(self):
        """Skips when present."""
        store, mock_client = _make_store()
        mock_client.collection_exists.return_value = True
        await store.ensure_collection("existing_col", 1536)
        mock_client.create_collection.assert_not_called()


class TestQdrantVectorStoreUpsert:
    """Tests for QdrantVectorStore.upsert."""

    async def test_delegates_correct_point_count(self):
        """Verify upsert sends correct number of points to the client."""
        store, mock_client = _make_store()
        points = [
            VectorRecord(id=_E1, vector=[0.1, 0.2], payload={"name": "BRCA1"}),
            VectorRecord(id=_E2, vector=[0.3, 0.4], payload={"name": "TP53"}),
        ]
        await store.upsert("entities", points)
        mock_client.upsert.assert_called_once()
        sent = mock_client.upsert.call_args.kwargs["points"]
        assert len(sent) == 2
        assert sent[0].id == _E1
        assert sent[1].payload == {"name": "TP53"}

    async def test_empty_list_does_not_call_client(self):
        """Verify upsert with empty list does not call the client."""
        store, mock_client = _make_store()
        await store.upsert("entities", [])
        mock_client.upsert.assert_not_called()

    async def test_batching_splits_into_multiple_calls(self):
        """Verify large point lists are split into multiple upsert calls."""
        store, mock_client = _make_store()
        points = [VectorRecord(id=uuid4(), vector=[float(i)]) for i in range(10)]
        await store.upsert("col", points, batch_size=4)
        # 10 points / 4 = 3 batches (4 + 4 + 2)
        assert mock_client.upsert.call_count == 3
        sizes = [
            len(call.kwargs["points"]) for call in mock_client.upsert.call_args_list
        ]
        assert sizes == [4, 4, 2]

    async def test_single_batch_when_points_le_batch_size(self):
        """Verify points under the batch size make a single upsert call."""
        store, mock_client = _make_store()
        points = [VectorRecord(id=uuid4(), vector=[float(i)]) for i in range(5)]
        await store.upsert("col", points, batch_size=256)
        assert mock_client.upsert.call_count == 1

    async def test_exact_multiple_of_batch_size(self):
        """Verify exact multiples of batch size produce correct call count."""
        store, mock_client = _make_store()
        points = [VectorRecord(id=uuid4(), vector=[float(i)]) for i in range(8)]
        await store.upsert("col", points, batch_size=4)
        assert mock_client.upsert.call_count == 2


class TestQdrantVectorStoreRetrieve:
    """Tests for QdrantVectorStore.retrieve."""

    def _make_record(
        self,
        id_: UUID,
        payload: dict | None,
        vector: list[float] | None = None,
    ):
        rec = MagicMock()
        rec.id = str(id_)
        rec.payload = payload
        rec.vector = vector
        return rec

    async def test_empty_ids_returns_empty_without_calling_client(self):
        """Verify empty ID list returns empty and does not call client."""
        store, mock_client = _make_store()
        result = await store.retrieve("entities", [])
        assert result == []
        mock_client.retrieve.assert_not_called()

    async def test_maps_records_to_vector_records(self):
        """Verify Qdrant records are mapped to VectorRecord instances."""
        store, mock_client = _make_store()
        mock_client.retrieve.return_value = [
            self._make_record(_E1, {"name": "BRCA1"}),
        ]
        result = await store.retrieve("entities", [str(_E1)])
        assert len(result) == 1
        assert result[0].id == _E1
        assert result[0].payload == {"name": "BRCA1"}
        assert result[0].vector == []

    async def test_with_vectors_true_populates_vector(self):
        """Verify with_vectors=True populates the vector field."""
        store, mock_client = _make_store()
        mock_client.retrieve.return_value = [
            self._make_record(_E1, {}, vector=[0.1, 0.2, 0.3]),
        ]
        result = await store.retrieve("entities", [str(_E1)], with_vectors=True)
        assert result[0].vector == [0.1, 0.2, 0.3]

    async def test_with_vectors_false_gives_empty_vector(self):
        """Verify with_vectors=False returns an empty vector."""
        store, mock_client = _make_store()
        mock_client.retrieve.return_value = [
            self._make_record(_E1, {}, vector=[0.1, 0.2]),
        ]
        result = await store.retrieve("entities", [str(_E1)], with_vectors=False)
        assert result[0].vector == []

    async def test_none_payload_becomes_empty_dict(self):
        """Verify None payload from Qdrant becomes an empty dict."""
        store, mock_client = _make_store()
        mock_client.retrieve.return_value = [self._make_record(_E1, None)]
        result = await store.retrieve("entities", [str(_E1)])
        assert result[0].payload == {}

    async def test_passes_with_vectors_flag_to_client(self):
        """Verify with_vectors flag is passed through to the Qdrant client."""
        store, mock_client = _make_store()
        mock_client.retrieve.return_value = []
        await store.retrieve("entities", [str(_E1)], with_vectors=True)
        call_kwargs = mock_client.retrieve.call_args.kwargs
        assert call_kwargs["with_vectors"] is True


class TestQdrantVectorStoreCount:
    """Tests for QdrantVectorStore.count."""

    async def test_returns_count_from_client(self):
        """Verify count returns the count from the client."""
        store, mock_client = _make_store()
        count_result = MagicMock()
        count_result.count = 42
        mock_client.count.return_value = count_result
        assert await store.count("entities") == 42

    async def test_requests_exact_count(self):
        """Verify count requests an exact count from the client."""
        store, mock_client = _make_store()
        count_result = MagicMock()
        count_result.count = 0
        mock_client.count.return_value = count_result
        await store.count("entities")
        mock_client.count.assert_called_once_with(
            collection_name="entities",
            exact=True,
        )


class TestQdrantVectorStoreSearch:
    """Tests for QdrantVectorStore.search."""

    async def test_returns_empty_on_no_hits(self):
        """Verify search returns empty when there are no hits."""
        store, mock_client = _make_store()
        mock_client.query_points.return_value = _query_response([])
        results = await store.search("entities", [0.1, 0.2], limit=5)
        assert results == []

    async def test_maps_hits_to_vector_hits(self):
        """Verify search results are mapped to VectorHit instances."""
        store, mock_client = _make_store()
        hit = MagicMock()
        hit.id = str(_E1)
        hit.score = 0.91
        hit.payload = {"name": "BRCA1"}
        mock_client.query_points.return_value = _query_response([hit])
        results = await store.search("entities", [0.1, 0.2])
        assert len(results) == 1
        assert isinstance(results[0], VectorHit)
        assert results[0].id == _E1
        assert results[0].score == 0.91

    async def test_passes_limit_and_threshold(self):
        """Verify limit and score_threshold are passed to the client."""
        store, mock_client = _make_store()
        mock_client.query_points.return_value = _query_response([])
        await store.search("col", [0.0], limit=3, score_threshold=0.8)
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 3
        assert call_kwargs["score_threshold"] == 0.8

    async def test_no_filter_passes_none_to_client(self):
        """Verify no filter passes None to the client."""
        store, mock_client = _make_store()
        mock_client.query_points.return_value = _query_response([])
        await store.search("col", [0.1])
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    async def test_scalar_filter_is_passed_to_client(self):
        """Verify scalar filter is passed to the client."""
        store, mock_client = _make_store()
        mock_client.query_points.return_value = _query_response([])
        await store.search("col", [0.1], filters={"label": "disease"})
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    async def test_list_filter_is_passed_to_client(self):
        """Verify list filter is passed to the client."""
        store, mock_client = _make_store()
        mock_client.query_points.return_value = _query_response([])
        await store.search(
            "col", [0.1], filters={"source": ["pubmed", "pubmed_case_reports"]}
        )
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    async def test_none_payload_becomes_empty_dict(self):
        """Verify None payload from search hit becomes empty dict."""
        store, mock_client = _make_store()
        hit = MagicMock()
        hit.id = str(_E1)
        hit.score = 0.5
        hit.payload = None
        mock_client.query_points.return_value = _query_response([hit])
        results = await store.search("col", [0.1])
        assert results[0].payload == {}


class TestQdrantVectorStoreHybridSearch:
    """Tests for QdrantVectorStore.hybrid_search."""

    async def test_falls_back_to_dense_search(self):
        """Verify hybrid_search falls back to dense search."""
        store, mock_client = _make_store()
        mock_client.query_points.return_value = _query_response([])
        results = await store.hybrid_search("col", [0.1, 0.2], "diabetes", limit=5)
        assert results == []
        mock_client.query_points.assert_called_once()


class TestQdrantVectorStoreScroll:
    """Tests for QdrantVectorStore.scroll."""

    def _make_record(
        self,
        id_: UUID,
        payload: dict | None,
        vector: list[float] | None = None,
    ):
        rec = MagicMock()
        rec.id = str(id_)
        rec.payload = payload
        rec.vector = vector
        return rec

    async def test_returns_empty_on_no_records(self):
        """Verify scroll returns empty when there are no records."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = ([], None)
        points, next_offset = await store.scroll("entities")
        assert points == []
        assert next_offset is None

    async def test_maps_records_to_vector_records(self):
        """Verify scroll records are mapped to VectorRecord instances."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = (
            [self._make_record(_E1, {"name": "BRCA1"})],
            None,
        )
        points, _ = await store.scroll("entities")
        assert len(points) == 1
        assert points[0].id == _E1
        assert points[0].payload == {"name": "BRCA1"}
        assert points[0].vector == []

    async def test_with_vectors_true_populates_vector(self):
        """Verify scroll with_vectors=True populates the vector field."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = (
            [self._make_record(_E1, {}, vector=[0.1, 0.2])],
            None,
        )
        points, _ = await store.scroll("entities", with_vectors=True)
        assert points[0].vector == [0.1, 0.2]

    async def test_next_offset_converted_to_string(self):
        """Verify the next offset is returned as a string."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = ([], "abc-123")
        _, next_offset = await store.scroll("entities")
        assert next_offset == "abc-123"

    async def test_none_next_offset_stays_none(self):
        """Verify None next offset is returned as None."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = ([], None)
        _, next_offset = await store.scroll("entities")
        assert next_offset is None

    async def test_passes_page_offset_to_client(self):
        """Verify page_offset is passed to the client."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = ([], None)
        await store.scroll("col", page_offset="cursor-xyz")
        call_kwargs = mock_client.scroll.call_args.kwargs
        assert call_kwargs["offset"] == "cursor-xyz"

    async def test_no_filter_passes_none_to_client(self):
        """Verify no filter passes None as scroll_filter."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = ([], None)
        await store.scroll("col")
        call_kwargs = mock_client.scroll.call_args.kwargs
        assert call_kwargs["scroll_filter"] is None

    async def test_filter_is_passed_to_client(self):
        """Verify filters are passed to the client."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = ([], None)
        await store.scroll("col", filters={"label": "entity"})
        call_kwargs = mock_client.scroll.call_args.kwargs
        assert call_kwargs["scroll_filter"] is not None

    async def test_none_payload_becomes_empty_dict(self):
        """Verify None payload from scroll becomes empty dict."""
        store, mock_client = _make_store()
        mock_client.scroll.return_value = (
            [self._make_record(_E1, None)],
            None,
        )
        points, _ = await store.scroll("col")
        assert points[0].payload == {}


class TestQdrantVectorStoreDelete:
    """Tests for QdrantVectorStore.delete."""

    async def test_empty_ids_returns_without_calling_client(self):
        """Verify empty ID list does not call the client."""
        store, mock_client = _make_store()
        await store.delete("entities", [])
        mock_client.delete.assert_not_called()

    async def test_passes_ids_to_client(self):
        """Verify IDs are passed to the client as points_selector."""
        store, mock_client = _make_store()
        await store.delete("entities", ["e1", "e2"])
        mock_client.delete.assert_called_once()
        selector = mock_client.delete.call_args.kwargs["points_selector"]
        assert selector.points == ["e1", "e2"]


class TestQdrantVectorStoreClose:
    """Tests for QdrantVectorStore.close."""

    async def test_close_delegates_to_client_and_nulls_it(self):
        """Verify close delegates to client and clears the reference."""
        store, mock_client = _make_store()
        await store.close()
        mock_client.close.assert_called_once()
        assert store._client is None

    async def test_close_is_noop_when_client_not_initialised(self):
        """Verify close is safe when client was never created."""
        store = QdrantVectorStore(_SETTINGS)
        await store.close()

    async def test_context_manager_closes_on_exit(self):
        """Verify async context manager closes the client on exit."""
        store, mock_client = _make_store()
        async with store:
            pass
        mock_client.close.assert_called_once()
        assert store._client is None


class TestCreateQdrantStore:
    """Tests for create_qdrant_store factory."""

    def test_returns_qdrant_vector_store(self):
        """Verify create_qdrant_store returns a QdrantVectorStore instance."""
        store = create_qdrant_store(_SETTINGS)
        assert isinstance(store, QdrantVectorStore)

    def test_client_not_created_eagerly(self):
        """Verify client is not created until first use (lazy init)."""
        store = create_qdrant_store(_SETTINGS)
        assert store._client is None

    async def test_uses_env_settings_when_none_provided(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Verify env settings are used when no settings argument is given."""
        monkeypatch.setenv("QDRANT_URL", "http://env-host:6333")
        with patch("am_diag.db.vector.qdrant.AsyncQdrantClient") as mock_cls:
            store = create_qdrant_store()
            await store._get_client()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["url"] == "http://env-host:6333"
