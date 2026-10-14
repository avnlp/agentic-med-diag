"""Unit tests for SentenceTransformersEmbedder with a mocked SentenceTransformer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from am_diag.vector.embedding.sentence_transformers import (
    SentenceTransformersEmbedder,
)


_PATCH_TARGET = "am_diag.vector.embedding.sentence_transformers.SentenceTransformer"


def _mock_model(vectors: list[list[float]]) -> MagicMock:
    """Build a mock SentenceTransformer whose .encode returns a numpy array."""
    model = MagicMock()
    model.encode = MagicMock(return_value=np.array(vectors, dtype=np.float32))
    return model


def _make_embedder(
    vectors: list[list[float]],
    **kwargs: object,
) -> tuple[SentenceTransformersEmbedder, MagicMock]:
    """Return an embedder with ``_model`` pre-set (bypasses lazy init)."""
    embedder = SentenceTransformersEmbedder(**kwargs)
    model = _mock_model(vectors)
    embedder._model = model
    return embedder, model


class TestSentenceTransformersEmbedderLazyInit:
    """Tests for lazy model initialisation."""

    def test_model_is_none_before_first_call(self):
        """Verify _model is None before first call."""
        embedder = SentenceTransformersEmbedder()
        assert embedder._model is None

    async def test_get_model_creates_model(self):
        """Verify _get_model creates a new model instance."""
        with patch(_PATCH_TARGET) as mock_cls:
            embedder = SentenceTransformersEmbedder(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
            )
            model = await embedder._get_model()
            mock_cls.assert_called_once_with(
                "all-MiniLM-L6-v2",
                device="cpu",
            )
            assert model is mock_cls.return_value

    async def test_get_model_reuses_existing(self):
        """Verify _get_model reuses an existing model instance."""
        with patch(_PATCH_TARGET) as mock_cls:
            embedder = SentenceTransformersEmbedder()
            m1 = await embedder._get_model()
            m2 = await embedder._get_model()
            mock_cls.assert_called_once()
            assert m1 is m2

    async def test_omits_trust_remote_code_and_model_kwargs_by_default(self):
        """Verify trust_remote_code is omitted by default."""
        with patch(_PATCH_TARGET) as mock_cls:
            embedder = SentenceTransformersEmbedder()
            await embedder._get_model()
            call_kwargs = mock_cls.call_args.kwargs
            assert "trust_remote_code" not in call_kwargs
            assert "model_kwargs" not in call_kwargs

    async def test_passes_trust_remote_code_and_torch_dtype_when_configured(self):
        """Verify trust_remote_code and torch_dtype are passed when set."""
        with patch(_PATCH_TARGET) as mock_cls:
            embedder = SentenceTransformersEmbedder(
                model_name="zeroentropy/zembed-1",
                device="cpu",
                trust_remote_code=True,
                torch_dtype="bfloat16",
            )
            await embedder._get_model()
            mock_cls.assert_called_once_with(
                "zeroentropy/zembed-1",
                device="cpu",
                trust_remote_code=True,
                model_kwargs={"torch_dtype": "bfloat16"},
            )


class TestSentenceTransformersEmbedderEmbed:
    """Tests for embed method."""

    async def test_returns_empty_list_for_empty_input(self):
        """Verify embedding zero texts returns empty without calling the model."""
        embedder, model = _make_embedder([])
        result = await embedder.embed([])
        assert result == []
        model.encode.assert_not_called()

    async def test_calls_encode_with_expected_kwargs(self):
        """Verify encode is called with expected keyword arguments."""
        embedder, model = _make_embedder(
            [[0.1, 0.2], [0.3, 0.4]],
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=32,
            normalize=True,
        )
        await embedder.embed(["alpha", "beta"])
        model.encode.assert_called_once()
        call_args, call_kwargs = model.encode.call_args
        assert call_args[0] == ["alpha", "beta"]
        assert call_kwargs["batch_size"] == 32
        assert call_kwargs["normalize_embeddings"] is True
        assert call_kwargs["convert_to_numpy"] is True
        assert call_kwargs["show_progress_bar"] is False

    async def test_returns_list_of_lists_of_floats(self):
        """Verify output is a list of lists of floats."""
        embedder, _ = _make_embedder([[0.1, 0.2], [0.3, 0.4]])
        result = await embedder.embed(["alpha", "beta"])
        assert isinstance(result, list)
        assert len(result) == 2
        for vector in result:
            assert isinstance(vector, list)
            for value in vector:
                assert isinstance(value, float)
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])

    async def test_preserves_input_order(self):
        """Verify output order matches input order."""
        embedder, _ = _make_embedder([[1.0], [2.0], [3.0]])
        result = await embedder.embed(["first", "second", "third"])
        assert result == [[1.0], [2.0], [3.0]]

    async def test_passes_custom_batch_size_and_normalize(self):
        """Verify custom batch_size and normalize are forwarded."""
        embedder, model = _make_embedder(
            [[0.5]],
            batch_size=4,
            normalize=False,
        )
        await embedder.embed(["x"])
        call_kwargs = model.encode.call_args.kwargs
        assert call_kwargs["batch_size"] == 4
        assert call_kwargs["normalize_embeddings"] is False


def _make_embedder_with_input_type(
    input_type: str,
) -> tuple[SentenceTransformersEmbedder, MagicMock]:
    """Return an embedder configured with the given ``input_type``."""
    embedder = SentenceTransformersEmbedder(
        model_name="some-model",
        input_type=input_type,
    )
    model = MagicMock()
    vectors = np.array([[0.1, 0.2]], dtype=np.float32)
    model.encode = MagicMock(return_value=vectors)
    model.encode_document = MagicMock(return_value=vectors)
    model.encode_query = MagicMock(return_value=vectors)
    embedder._model = model
    return embedder, model


class TestSentenceTransformersEmbedderInputType:
    """Tests for input_type parameter dispatch."""

    async def test_generic_input_type_uses_encode(self):
        """Verify generic input type uses encode method."""
        embedder, model = _make_embedder_with_input_type("generic")
        await embedder.embed(["x"])
        model.encode.assert_called_once()
        model.encode_document.assert_not_called()
        model.encode_query.assert_not_called()

    async def test_document_input_type_uses_encode_document(self):
        """Verify document input type uses encode_document."""
        embedder, model = _make_embedder_with_input_type("document")
        await embedder.embed(["x"])
        model.encode_document.assert_called_once()
        model.encode.assert_not_called()
        model.encode_query.assert_not_called()

    async def test_query_input_type_uses_encode_query(self):
        """Verify query input type uses encode_query."""
        embedder, model = _make_embedder_with_input_type("query")
        await embedder.embed(["x"])
        model.encode_query.assert_called_once()
        model.encode.assert_not_called()
        model.encode_document.assert_not_called()


class TestSentenceTransformersEmbedderPrecisionAndDimensions:
    """Tests for precision and dimension handling."""

    async def test_float32_precision_is_omitted(self):
        """Verify float32 precision is omitted from encode kwargs."""
        embedder, model = _make_embedder([[0.1, 0.2]])
        await embedder.embed(["x"])
        assert "precision" not in model.encode.call_args.kwargs

    async def test_non_float32_precision_is_passed(self):
        """Verify non-float32 precision is passed to encode."""
        embedder, model = _make_embedder(
            [[0.1, 0.2]],
            precision="int8",
        )
        await embedder.embed(["x"])
        assert model.encode.call_args.kwargs["precision"] == "int8"

    async def test_dimensions_maps_to_truncate_dim(self):
        """Verify dimensions maps to truncate_dim in encode kwargs."""
        embedder, model = _make_embedder(
            [[0.1, 0.2]],
            dimensions=128,
        )
        await embedder.embed(["x"])
        assert model.encode.call_args.kwargs["truncate_dim"] == 128

    async def test_no_dimensions_omits_truncate_dim(self):
        """Verify truncate_dim is omitted when dimensions is None."""
        embedder, model = _make_embedder([[0.1, 0.2]])
        await embedder.embed(["x"])
        assert "truncate_dim" not in model.encode.call_args.kwargs


class TestSentenceTransformersEmbedderClose:
    """Tests for close / async context manager."""

    async def test_close_clears_model_reference(self):
        """Verify close clears the internal model reference."""
        embedder, _ = _make_embedder([[0.1]])
        assert embedder._model is not None
        await embedder.close()
        assert embedder._model is None

    async def test_close_is_safe_when_model_never_loaded(self):
        """Verify close is safe when model was never loaded."""
        embedder = SentenceTransformersEmbedder()
        await embedder.close()
        assert embedder._model is None

    async def test_async_context_manager_calls_close(self):
        """Verify async context manager calls close on exit."""
        embedder, _ = _make_embedder([[0.1]])
        async with embedder as ctx:
            assert ctx is embedder
        assert embedder._model is None


class TestSentenceTransformersEmbedderModelId:
    """Tests for the model_id cache-key identity."""

    def test_model_id_is_model_name_without_dimensions(self):
        """Without a truncation dimension, model_id is just the model name."""
        embedder = SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_id == "all-MiniLM-L6-v2"

    def test_model_id_includes_dimension_when_set(self):
        """A truncation dimension is part of the identity to avoid cache reuse."""
        embedder = SentenceTransformersEmbedder(
            model_name="all-MiniLM-L6-v2", dimensions=128
        )
        assert embedder.model_id == "all-MiniLM-L6-v2@128"
