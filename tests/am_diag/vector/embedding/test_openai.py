"""Unit tests for OpenAIEmbedder with a mocked AsyncOpenAI client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from am_diag.vector.embedding.openai import OpenAIEmbedder


_PATCH_TARGET = "am_diag.vector.embedding.openai.AsyncOpenAI"


def _embedding_item(vector: list[float]) -> MagicMock:
    item = MagicMock()
    item.embedding = vector
    return item


def _embedding_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [_embedding_item(v) for v in vectors]
    return response


def _make_embedder(
    responses: list[MagicMock] | None = None,
    **kwargs: object,
) -> tuple[OpenAIEmbedder, MagicMock]:
    """Return an embedder with ``_client`` pre-set to a mock (bypasses lazy init)."""
    mock_client = MagicMock()
    mock_client.embeddings = MagicMock()
    if responses is not None:
        mock_client.embeddings.create = AsyncMock(side_effect=responses)
    else:
        mock_client.embeddings.create = AsyncMock()
    mock_client.close = AsyncMock()

    embedder = OpenAIEmbedder(**kwargs)
    embedder._client = mock_client  # inject: bypass lazy _get_client()
    return embedder, mock_client


class TestOpenAIEmbedderLazyInit:
    def test_client_is_none_before_first_call(self):
        embedder = OpenAIEmbedder()
        assert embedder._client is None

    async def test_get_client_creates_client(self):
        with patch(_PATCH_TARGET) as mock_cls:
            embedder = OpenAIEmbedder()
            client = await embedder._get_client()
            mock_cls.assert_called_once_with()
            assert client is mock_cls.return_value

    async def test_get_client_reuses_existing(self):
        with patch(_PATCH_TARGET) as mock_cls:
            embedder = OpenAIEmbedder()
            c1 = await embedder._get_client()
            c2 = await embedder._get_client()
            mock_cls.assert_called_once()
            assert c1 is c2


class TestOpenAIEmbedderEmbed:
    async def test_returns_empty_list_for_empty_input(self):
        embedder, mock_client = _make_embedder()
        result = await embedder.embed([])
        assert result == []
        mock_client.embeddings.create.assert_not_called()

    async def test_calls_embeddings_create_with_model_and_input(self):
        embedder, mock_client = _make_embedder(
            [_embedding_response([[0.1, 0.2]])],
            model_name="text-embedding-3-small",
        )
        await embedder.embed(["alpha"])
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["alpha"],
        )

    async def test_returns_list_of_lists_of_floats(self):
        embedder, _ = _make_embedder(
            [_embedding_response([[0.1, 0.2], [0.3, 0.4]])],
        )
        result = await embedder.embed(["alpha", "beta"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    async def test_chunks_requests_by_batch_size(self):
        """batch_size=2; three texts -> two calls: [a, b] then [c]."""
        embedder, mock_client = _make_embedder(
            [
                _embedding_response([[0.1], [0.2]]),
                _embedding_response([[0.3]]),
            ],
            batch_size=2,
        )
        result = await embedder.embed(["a", "b", "c"])
        assert mock_client.embeddings.create.call_count == 2
        first_call, second_call = mock_client.embeddings.create.call_args_list
        assert first_call.kwargs["input"] == ["a", "b"]
        assert second_call.kwargs["input"] == ["c"]
        assert result == [[0.1], [0.2], [0.3]]

    async def test_preserves_order_across_batches(self):
        embedder, _ = _make_embedder(
            [
                _embedding_response([[1.0], [2.0]]),
                _embedding_response([[3.0], [4.0]]),
            ],
            batch_size=2,
        )
        result = await embedder.embed(["w", "x", "y", "z"])
        assert result == [[1.0], [2.0], [3.0], [4.0]]


class TestOpenAIEmbedderClose:
    async def test_close_delegates_to_client_and_clears_reference(self):
        embedder, mock_client = _make_embedder()
        await embedder.close()
        mock_client.close.assert_called_once()
        assert embedder._client is None

    async def test_close_is_safe_when_client_never_created(self):
        embedder = OpenAIEmbedder()
        await embedder.close()
        assert embedder._client is None

    async def test_async_context_manager_calls_close(self):
        embedder, mock_client = _make_embedder()
        async with embedder as ctx:
            assert ctx is embedder
        mock_client.close.assert_called_once()
        assert embedder._client is None
