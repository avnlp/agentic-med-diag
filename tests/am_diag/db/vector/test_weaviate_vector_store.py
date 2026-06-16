"""Unit tests for WeaviateVectorStore with mocked Weaviate client."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from am_diag.common.data_models.vector_record import VectorHit, VectorRecord
from am_diag.db.vector.settings import WeaviateSettings
from am_diag.db.vector.weaviate import (
    WeaviateVectorStore,
    _build_filter,
    _to_collection_name,
    _to_weaviate_uuid,
    create_weaviate_store,
)


_SETTINGS = WeaviateSettings(
    url="http://localhost:8080",
    api_key="",
    grpc_port=50051,
    batch_size=200,
)

# Stable UUIDs for tests
_E1 = uuid4()
_E2 = uuid4()


def _make_obj(
    original_id: str,
    payload: dict,
    distance: float = 0.1,
    vector: list[float] | None = None,
):
    """Build a minimal mock Weaviate result object."""
    obj = MagicMock()
    obj.properties = {
        "original_id": original_id,
        "payload": json.dumps(payload),
    }
    obj.metadata = MagicMock()
    obj.metadata.distance = distance
    obj.metadata.score = 1.0 - distance
    obj.uuid = uuid4()
    obj.vector = {"default": vector} if vector is not None else None
    return obj


def _make_store(
    settings: WeaviateSettings | None = None,
) -> tuple[WeaviateVectorStore, MagicMock, MagicMock]:
    """Return (store, mock_client, mock_collection) with client pre-injected."""
    mock_collection = MagicMock()
    mock_collection.query.near_vector = AsyncMock(return_value=MagicMock(objects=[]))
    mock_collection.query.hybrid = AsyncMock(return_value=MagicMock(objects=[]))
    mock_collection.query.fetch_objects = AsyncMock(return_value=MagicMock(objects=[]))
    mock_collection.data.insert_many = AsyncMock()
    mock_collection.data.delete_many = AsyncMock()
    mock_collection.aggregate.over_all = AsyncMock(
        return_value=MagicMock(total_count=0),
    )

    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.collections.exists = AsyncMock(return_value=False)
    mock_client.collections.create = AsyncMock()
    mock_client.collections.delete = AsyncMock()
    mock_client.collections.get.return_value = mock_collection

    store = WeaviateVectorStore(settings or _SETTINGS)
    store._client = mock_client  # bypass lazy connection in unit tests
    return store, mock_client, mock_collection


class TestToCollectionName:
    def test_single_word_capitalised(self):
        assert _to_collection_name("entities") == "Entities"

    def test_snake_case_becomes_pascal_case(self):
        assert _to_collection_name("text_chunks") == "TextChunks"

    def test_kebab_case_becomes_pascal_case(self):
        assert _to_collection_name("knowledge-graph") == "KnowledgeGraph"

    def test_already_pascal_case_unchanged(self):
        assert _to_collection_name("Entities") == "Entities"

    def test_multi_word_snake(self):
        assert _to_collection_name("my_vector_store") == "MyVectorStore"


class TestToWeaviateUuid:
    def test_deterministic_same_input(self):
        assert _to_weaviate_uuid("entity:disease:C001") == _to_weaviate_uuid(
            "entity:disease:C001",
        )

    def test_different_inputs_produce_different_uuids(self):
        assert _to_weaviate_uuid("a") != _to_weaviate_uuid("b")

    def test_returns_valid_uuid_string(self):
        result = _to_weaviate_uuid("test-id")
        parsed = uuid.UUID(result)  # raises if not valid UUID
        assert str(parsed) == result


class TestBuildFilter:
    def test_none_returns_none(self):
        assert _build_filter(None) is None

    def test_empty_dict_returns_none(self):
        assert _build_filter({}) is None

    def test_scalar_value_produces_filter(self):
        f = _build_filter({"type": "disease"})
        assert f is not None

    def test_list_value_produces_filter(self):
        f = _build_filter({"source": ["pubmed", "pubmed_case_reports"]})
        assert f is not None

    def test_multiple_keys_produces_filter(self):
        f = _build_filter({"type": "entity", "source": "pubmed"})
        assert f is not None

    def test_bool_scalar_produces_filter(self):
        f = _build_filter({"verified": True})
        assert f is not None

    def test_single_key_returns_single_filter(self):
        f1 = _build_filter({"type": "x"})
        f2 = _build_filter({"type": "x"})
        assert f1 is not None
        assert f2 is not None


class TestWeaviateVectorStoreCreateCollection:
    async def test_creates_when_not_exists(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = False
        await store.create_collection("entities", vector_size=1536)
        mock_client.collections.create.assert_called_once()

    async def test_skips_when_already_exists(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = True
        await store.create_collection("entities", vector_size=1536)
        mock_client.collections.create.assert_not_called()

    async def test_uses_pascal_collection_name(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = False
        await store.create_collection("text_chunks", vector_size=768)
        call_args = mock_client.collections.exists.call_args.args[0]
        assert call_args == "TextChunks"

    async def test_properties_include_original_id_and_payload(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = False
        await store.create_collection("entities", vector_size=128)
        call_kwargs = mock_client.collections.create.call_args.kwargs
        prop_names = [p.name for p in call_kwargs["properties"]]
        assert "original_id" in prop_names
        assert "payload" in prop_names


class TestWeaviateVectorStoreDeleteCollection:
    async def test_deletes_existing_collection(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = True
        await store.delete_collection("entities")
        mock_client.collections.delete.assert_called_once_with("Entities")

    async def test_noop_when_collection_missing(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = False
        await store.delete_collection("entities")
        mock_client.collections.delete.assert_not_called()


class TestWeaviateVectorStoreCollectionExists:
    async def test_returns_true_when_exists(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = True
        assert await store.collection_exists("entities") is True

    async def test_returns_false_when_missing(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = False
        assert await store.collection_exists("entities") is False


class TestWeaviateVectorStoreEnsureCollection:
    async def test_creates_when_absent(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = False
        await store.ensure_collection("new_col", vector_size=1536)
        mock_client.collections.create.assert_called_once()

    async def test_skips_when_present(self):
        store, mock_client, _ = _make_store()
        mock_client.collections.exists.return_value = True
        await store.ensure_collection("existing_col", vector_size=1536)
        mock_client.collections.create.assert_not_called()


class TestWeaviateVectorStoreUpsert:
    async def test_empty_list_does_not_call_client(self):
        store, _, mock_collection = _make_store()
        await store.upsert("entities", [])
        mock_collection.data.insert_many.assert_not_called()
        mock_collection.data.delete_many.assert_not_called()

    async def test_delete_then_insert_for_upsert_semantics(self):
        store, _, mock_collection = _make_store()
        points = [VectorRecord(id=_E1, vector=[0.1, 0.2], payload={"name": "BRCA1"})]
        await store.upsert("entities", points)
        mock_collection.data.delete_many.assert_called_once()
        mock_collection.data.insert_many.assert_called_once()

    async def test_insert_many_receives_correct_count(self):
        store, _, mock_collection = _make_store()
        points = [
            VectorRecord(id=_E1, vector=[0.1, 0.2], payload={"name": "BRCA1"}),
            VectorRecord(id=_E2, vector=[0.3, 0.4], payload={"name": "TP53"}),
        ]
        await store.upsert("entities", points)
        inserted = mock_collection.data.insert_many.call_args.args[0]
        assert len(inserted) == 2

    async def test_objects_carry_original_id_in_properties(self):
        store, _, mock_collection = _make_store()
        points = [VectorRecord(id=_E1, vector=[0.5], payload={"x": 1})]
        await store.upsert("col", points)
        inserted = mock_collection.data.insert_many.call_args.args[0]
        assert inserted[0].properties["original_id"] == str(_E1)

    async def test_payload_is_json_encoded(self):
        store, _, mock_collection = _make_store()
        payload = {"name": "BRCA1", "type": "gene"}
        points = [VectorRecord(id=_E1, vector=[0.1], payload=payload)]
        await store.upsert("col", points)
        inserted = mock_collection.data.insert_many.call_args.args[0]
        assert json.loads(inserted[0].properties["payload"]) == payload

    async def test_batching_into_multiple_calls(self):
        store, _, mock_collection = _make_store()
        points = [VectorRecord(id=uuid4(), vector=[float(i)]) for i in range(10)]
        await store.upsert("col", points, batch_size=4)
        # 10 points → 3 batches: 4 + 4 + 2
        assert mock_collection.data.insert_many.call_count == 3
        assert mock_collection.data.delete_many.call_count == 3

    async def test_batch_sizes_are_correct(self):
        store, _, mock_collection = _make_store()
        points = [VectorRecord(id=uuid4(), vector=[float(i)]) for i in range(10)]
        await store.upsert("col", points, batch_size=4)
        sizes = [
            len(call.args[0])
            for call in mock_collection.data.insert_many.call_args_list
        ]
        assert sizes == [4, 4, 2]

    async def test_exact_multiple_of_batch_size(self):
        store, _, mock_collection = _make_store()
        points = [VectorRecord(id=uuid4(), vector=[float(i)]) for i in range(8)]
        await store.upsert("col", points, batch_size=4)
        assert mock_collection.data.insert_many.call_count == 2

    async def test_single_batch_when_points_le_batch_size(self):
        store, _, mock_collection = _make_store()
        points = [VectorRecord(id=uuid4(), vector=[float(i)]) for i in range(5)]
        await store.upsert("col", points, batch_size=256)
        assert mock_collection.data.insert_many.call_count == 1

    async def test_uses_pascal_collection_name(self):
        store, mock_client, _ = _make_store()
        points = [VectorRecord(id=_E1, vector=[0.1])]
        await store.upsert("text_chunks", points)
        mock_client.collections.get.assert_called_once_with("TextChunks")


class TestWeaviateVectorStoreRetrieve:
    async def test_empty_ids_returns_empty(self):
        store, _, mock_collection = _make_store()
        result = await store.retrieve("entities", [])
        assert result == []
        mock_collection.query.fetch_objects.assert_not_called()

    async def test_maps_objects_to_vector_records(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(
            objects=[_make_obj(str(_E1), {"name": "BRCA1"})],
        )
        result = await store.retrieve("entities", [str(_E1)])
        assert len(result) == 1
        assert result[0].id == _E1
        assert result[0].payload == {"name": "BRCA1"}

    async def test_vector_is_empty_when_not_requested(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(
            objects=[_make_obj(str(_E1), {}, vector=[0.1, 0.2])],
        )
        result = await store.retrieve("entities", [str(_E1)], with_vectors=False)
        assert result[0].vector == []

    async def test_vector_populated_when_requested(self):
        store, _, mock_collection = _make_store()
        obj = _make_obj(str(_E1), {}, vector=[0.1, 0.2, 0.3])
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[obj])
        result = await store.retrieve("entities", [str(_E1)], with_vectors=True)
        assert result[0].vector == [0.1, 0.2, 0.3]

    async def test_with_vectors_flag_passed_to_client(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[])
        await store.retrieve("entities", [str(_E1)], with_vectors=True)
        call_kwargs = mock_collection.query.fetch_objects.call_args.kwargs
        assert call_kwargs["include_vector"] is True

    async def test_no_vector_field_gives_empty_list(self):
        store, _, mock_collection = _make_store()
        obj = _make_obj(str(_E1), {})
        obj.vector = None
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[obj])
        result = await store.retrieve("entities", [str(_E1)], with_vectors=True)
        assert result[0].vector == []


class TestWeaviateVectorStoreCount:
    async def test_returns_total_count(self):
        store, _, mock_collection = _make_store()
        mock_collection.aggregate.over_all.return_value = MagicMock(total_count=42)
        assert await store.count("entities") == 42

    async def test_none_total_count_returns_zero(self):
        store, _, mock_collection = _make_store()
        mock_collection.aggregate.over_all.return_value = MagicMock(total_count=None)
        assert await store.count("entities") == 0

    async def test_requests_total_count_from_aggregate(self):
        store, _, mock_collection = _make_store()
        await store.count("entities")
        mock_collection.aggregate.over_all.assert_called_once_with(total_count=True)


class TestWeaviateVectorStoreSearch:
    async def test_returns_empty_on_no_hits(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])
        results = await store.search("entities", [0.1, 0.2], limit=5)
        assert results == []

    async def test_maps_hits_to_vector_hits(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.near_vector.return_value = MagicMock(
            objects=[_make_obj(str(_E1), {"name": "BRCA1"}, distance=0.1)],
        )
        results = await store.search("entities", [0.1, 0.2])
        assert len(results) == 1
        assert isinstance(results[0], VectorHit)
        assert results[0].id == _E1

    async def test_score_is_one_minus_distance(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.near_vector.return_value = MagicMock(
            objects=[_make_obj(str(_E1), {}, distance=0.25)],
        )
        results = await store.search("entities", [0.1])
        assert abs(results[0].score - 0.75) < 1e-9

    async def test_payload_decoded_from_json(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.near_vector.return_value = MagicMock(
            objects=[_make_obj(str(_E1), {"key": "val"}, distance=0.0)],
        )
        results = await store.search("entities", [0.1])
        assert results[0].payload == {"key": "val"}

    async def test_passes_limit_to_client(self):
        store, _, mock_collection = _make_store()
        await store.search("col", [0.1], limit=7)
        call_kwargs = mock_collection.query.near_vector.call_args.kwargs
        assert call_kwargs["limit"] == 7

    async def test_no_filter_passes_none_to_client(self):
        store, _, mock_collection = _make_store()
        await store.search("col", [0.1])
        call_kwargs = mock_collection.query.near_vector.call_args.kwargs
        assert call_kwargs["filters"] is None

    async def test_filter_is_passed_to_client(self):
        store, _, mock_collection = _make_store()
        await store.search("col", [0.1], filters={"type": "disease"})
        call_kwargs = mock_collection.query.near_vector.call_args.kwargs
        assert call_kwargs["filters"] is not None

    async def test_score_threshold_filters_low_scoring_results(self):
        store, _, mock_collection = _make_store()
        e_high = uuid4()
        e_low = uuid4()
        mock_collection.query.near_vector.return_value = MagicMock(
            objects=[
                _make_obj(str(e_high), {}, distance=0.1),  # score=0.9
                _make_obj(str(e_low), {}, distance=0.8),  # score=0.2
            ],
        )
        results = await store.search("col", [0.1], score_threshold=0.5)
        assert len(results) == 1
        assert results[0].id == e_high

    async def test_score_threshold_none_returns_all(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.near_vector.return_value = MagicMock(
            objects=[
                _make_obj(str(_E1), {}, distance=0.1),
                _make_obj(str(_E2), {}, distance=0.9),
            ],
        )
        results = await store.search("col", [0.1], score_threshold=None)
        assert len(results) == 2

    async def test_uses_pascal_collection_name(self):
        store, mock_client, _ = _make_store()
        await store.search("text_chunks", [0.1])
        mock_client.collections.get.assert_called_once_with("TextChunks")


class TestWeaviateVectorStoreHybridSearch:
    async def test_uses_native_hybrid_query(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.hybrid.return_value = MagicMock(objects=[])
        results = await store.hybrid_search(
            "entities", [0.1, 0.2], "diabetes", limit=5, alpha=0.7
        )
        assert results == []
        mock_collection.query.hybrid.assert_called_once()

    async def test_passes_alpha_to_client(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.hybrid.return_value = MagicMock(objects=[])
        await store.hybrid_search("col", [0.1], "query", alpha=0.3)
        call_kwargs = mock_collection.query.hybrid.call_args.kwargs
        assert call_kwargs["alpha"] == 0.3

    async def test_passes_query_text_to_client(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.hybrid.return_value = MagicMock(objects=[])
        await store.hybrid_search("col", [0.1], "diabetes query", alpha=0.5)
        call_kwargs = mock_collection.query.hybrid.call_args.kwargs
        assert call_kwargs["query"] == "diabetes query"

    async def test_maps_results_to_vector_hits(self):
        store, _, mock_collection = _make_store()
        obj = _make_obj(str(_E1), {"name": "BRCA1"}, distance=0.1)
        obj.metadata.score = 0.9
        mock_collection.query.hybrid.return_value = MagicMock(objects=[obj])
        results = await store.hybrid_search("col", [0.1], "query", alpha=0.5)
        assert len(results) == 1
        assert isinstance(results[0], VectorHit)
        assert results[0].id == _E1


class TestWeaviateVectorStoreScroll:
    async def test_empty_collection_returns_none_offset(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[])
        points, next_offset = await store.scroll("entities")
        assert points == []
        assert next_offset is None

    async def test_maps_objects_to_vector_records(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(
            objects=[_make_obj(str(_E1), {"name": "BRCA1"})],
        )
        points, _ = await store.scroll("entities")
        assert len(points) == 1
        assert points[0].id == _E1
        assert points[0].payload == {"name": "BRCA1"}
        assert points[0].vector == []

    async def test_full_page_provides_next_offset(self):
        store, _, mock_collection = _make_store()
        objs = [_make_obj(str(uuid4()), {}) for _ in range(3)]
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=objs)
        _, next_offset = await store.scroll("entities", limit=3)
        assert next_offset == str(objs[-1].uuid)

    async def test_partial_page_returns_none_offset(self):
        store, _, mock_collection = _make_store()
        objs = [_make_obj(str(uuid4()), {}) for _ in range(2)]
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=objs)
        _, next_offset = await store.scroll("entities", limit=5)
        assert next_offset is None

    async def test_page_offset_passed_to_client_as_after(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[])
        await store.scroll("col", page_offset="some-cursor-uuid")
        call_kwargs = mock_collection.query.fetch_objects.call_args.kwargs
        assert call_kwargs["after"] == "some-cursor-uuid"

    async def test_none_offset_starts_from_beginning(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[])
        await store.scroll("col", page_offset=None)
        call_kwargs = mock_collection.query.fetch_objects.call_args.kwargs
        assert call_kwargs["after"] is None

    async def test_no_filter_passes_none_to_client(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[])
        await store.scroll("col")
        call_kwargs = mock_collection.query.fetch_objects.call_args.kwargs
        assert call_kwargs["filters"] is None

    async def test_filter_is_passed_to_client(self):
        store, _, mock_collection = _make_store()
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[])
        await store.scroll("col", filters={"type": "entity"})
        call_kwargs = mock_collection.query.fetch_objects.call_args.kwargs
        assert call_kwargs["filters"] is not None

    async def test_with_vectors_populates_vector(self):
        store, _, mock_collection = _make_store()
        obj = _make_obj(str(_E1), {}, vector=[0.1, 0.2])
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[obj])
        points, _ = await store.scroll("entities", limit=10, with_vectors=True)
        assert points[0].vector == [0.1, 0.2]

    async def test_with_vectors_false_gives_empty_vector(self):
        store, _, mock_collection = _make_store()
        obj = _make_obj(str(_E1), {}, vector=[0.1, 0.2])
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[obj])
        points, _ = await store.scroll("entities", limit=10, with_vectors=False)
        assert points[0].vector == []


class TestWeaviateVectorStoreDelete:
    async def test_empty_ids_does_not_call_client(self):
        store, _, mock_collection = _make_store()
        await store.delete("entities", [])
        mock_collection.data.delete_many.assert_not_called()

    async def test_calls_delete_many_with_uuids(self):
        store, _, mock_collection = _make_store()
        await store.delete("entities", [str(_E1), str(_E2)])
        mock_collection.data.delete_many.assert_called_once()

    async def test_uses_pascal_collection_name(self):
        store, mock_client, _ = _make_store()
        await store.delete("text_chunks", [str(_E1)])
        mock_client.collections.get.assert_called_once_with("TextChunks")


class TestWeaviateVectorStoreClose:
    async def test_close_calls_client_close(self):
        store, mock_client, _ = _make_store()
        await store.close()
        mock_client.close.assert_called_once()

    async def test_close_sets_client_to_none(self):
        store, _, _ = _make_store()
        await store.close()
        assert store._client is None

    async def test_close_when_not_connected_is_noop(self):
        store = WeaviateVectorStore(_SETTINGS)
        # _client is None — should not raise
        await store.close()

    async def test_context_manager_closes_on_exit(self):
        store, mock_client, _ = _make_store()
        async with store:
            pass
        mock_client.close.assert_called_once()


class TestWeaviateVectorStoreLazyConnect:
    async def test_connect_triggered_on_first_use(self):
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.collections.exists = AsyncMock(return_value=False)

        with patch(
            "am_diag.db.vector.weaviate._connect",
            new=AsyncMock(return_value=mock_client),
        ) as mock_connect:
            store = WeaviateVectorStore(_SETTINGS)
            assert store._client is None
            await store.collection_exists("entities")
            mock_connect.assert_called_once()

    async def test_second_call_reuses_connection(self):
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.collections.exists = AsyncMock(return_value=False)

        with patch(
            "am_diag.db.vector.weaviate._connect",
            new=AsyncMock(return_value=mock_client),
        ) as mock_connect:
            store = WeaviateVectorStore(_SETTINGS)
            await store.collection_exists("a")
            await store.collection_exists("b")
            mock_connect.assert_called_once()  # only one connection


class TestCreateWeaviateStore:
    def test_returns_weaviate_vector_store(self):
        store = create_weaviate_store(_SETTINGS)
        assert isinstance(store, WeaviateVectorStore)

    def test_uses_env_settings_when_none_provided(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("WEAVIATE_URL", "http://env-host:8080")
        store = create_weaviate_store()
        assert store._settings.url == "http://env-host:8080"

    def test_explicit_settings_override_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("WEAVIATE_URL", "http://ignored")
        settings = WeaviateSettings(url="http://explicit:8080")
        store = create_weaviate_store(settings)
        assert store._settings.url == "http://explicit:8080"
