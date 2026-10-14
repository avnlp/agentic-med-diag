"""Unit tests for SentenceTransformersReranker with a mocked CrossEncoder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from am_diag.vector.reranking.base import RerankResult
from am_diag.vector.reranking.sentence_transformers import (
    SentenceTransformersReranker,
)


_PATCH_TARGET = "am_diag.vector.reranking.sentence_transformers.CrossEncoder"


def _mock_cross_encoder(raw_scores: list[float]) -> MagicMock:
    """Build a mock CrossEncoder whose .predict returns raw logits."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array(raw_scores, dtype=np.float32))
    return model


def _make_reranker(
    raw_scores: list[float],
    **kwargs: object,
) -> tuple[SentenceTransformersReranker, MagicMock]:
    """Return a reranker with ``_model`` pre-set (bypasses lazy init)."""
    reranker = SentenceTransformersReranker(**kwargs)
    model = _mock_cross_encoder(raw_scores)
    reranker._model = model
    return reranker, model


class TestSentenceTransformersRerankerLazyInit:
    """Tests for lazy model initialisation."""

    def test_model_is_none_before_first_call(self):
        """Verify _model is None before first call."""
        reranker = SentenceTransformersReranker()
        assert reranker._model is None

    async def test_get_model_creates_model(self):
        """Verify _get_model creates a new model instance."""
        with patch(_PATCH_TARGET) as mock_cls:
            reranker = SentenceTransformersReranker(
                model_name="zerank-2-reranker",
                device="cpu",
                max_length=32768,
            )
            model = await reranker._get_model()
            mock_cls.assert_called_once_with(
                "zerank-2-reranker",
                device="cpu",
                max_length=32768,
            )
            assert model is mock_cls.return_value

    async def test_get_model_reuses_existing(self):
        """Verify _get_model reuses existing model instance."""
        with patch(_PATCH_TARGET) as mock_cls:
            reranker = SentenceTransformersReranker()
            m1 = await reranker._get_model()
            m2 = await reranker._get_model()
            mock_cls.assert_called_once()
            assert m1 is m2

    async def test_omits_trust_remote_code_and_torch_dtype_by_default(self):
        """Verify trust_remote_code and torch_dtype are omitted by default."""
        with patch(_PATCH_TARGET) as mock_cls:
            reranker = SentenceTransformersReranker()
            await reranker._get_model()
            call_kwargs = mock_cls.call_args.kwargs
            assert "trust_remote_code" not in call_kwargs
            assert "torch_dtype" not in call_kwargs

    async def test_passes_trust_remote_code_and_torch_dtype_when_configured(self):
        """Verify trust_remote_code and torch_dtype are passed when set."""
        with patch(_PATCH_TARGET) as mock_cls:
            reranker = SentenceTransformersReranker(
                model_name="zerank-2-reranker",
                device="cpu",
                trust_remote_code=True,
                torch_dtype="bfloat16",
            )
            await reranker._get_model()
            mock_cls.assert_called_once_with(
                "zerank-2-reranker",
                device="cpu",
                max_length=32768,
                trust_remote_code=True,
                torch_dtype="bfloat16",
            )


def _to_calibrated(raw_scores: list[float]) -> list[float]:
    import torch

    return (torch.tensor(raw_scores) / 5.0).sigmoid().tolist()


class TestSentenceTransformersRerankerRerank:
    """Tests for rerank method."""

    async def test_returns_empty_list_for_empty_documents(self):
        """Verify reranking zero documents skips predict and returns empty."""
        reranker, model = _make_reranker([])
        result = await reranker.rerank("q", [])
        assert result == []
        model.predict.assert_not_called()

    async def test_returns_empty_list_for_empty_query(self):
        """Verify empty query returns zero-score results."""
        reranker, model = _make_reranker([0.5, 0.6])
        result = await reranker.rerank("", ["doc a", "doc b"])
        assert len(result) == 2
        assert all(r.score == 0.0 for r in result)
        model.predict.assert_not_called()

    async def test_returns_empty_list_for_blank_query(self):
        """Verify blank query returns zero-score results."""
        reranker, model = _make_reranker([0.5, 0.6])
        result = await reranker.rerank("   ", ["doc a", "doc b"])
        assert len(result) == 2
        assert all(r.score == 0.0 for r in result)
        model.predict.assert_not_called()

    async def test_calls_predict_with_pairs(self):
        """Verify predict is called with query-document pairs."""
        reranker, model = _make_reranker([2.0, -1.0])
        await reranker.rerank("my query", ["alpha", "beta"])
        model.predict.assert_called_once()
        call_args, call_kwargs = model.predict.call_args
        assert call_args[0] == [("my query", "alpha"), ("my query", "beta")]
        assert call_kwargs["batch_size"] == 32

    async def test_calls_predict_with_instruction(self):
        """Verify instruction is appended to the query in predict pairs."""
        reranker, model = _make_reranker([2.0, -1.0])
        await reranker.rerank(
            "my query",
            ["alpha", "beta"],
            instruction="Prioritise medical accuracy",
        )
        model.predict.assert_called_once()
        call_args = model.predict.call_args
        expected_query = (
            "my query\n<instruction>Prioritise medical accuracy</instruction>"
        )
        pairs = call_args[0][0]
        assert pairs[0] == (expected_query, "alpha")
        assert pairs[1] == (expected_query, "beta")

    async def test_returns_rerank_results_with_calibrated_scores(self):
        """Verify raw scores are calibrated via sigmoid."""
        raw_scores = [5.0, 0.0, -5.0]
        reranker, _ = _make_reranker(raw_scores)
        results = await reranker.rerank("q", ["doc a", "doc b", "doc c"], top_k=None)
        expected = _to_calibrated(raw_scores)
        assert len(results) == 3
        for r, e in zip(results, expected, strict=False):
            assert r.score == pytest.approx(e, abs=1e-6)

    async def test_orders_results_by_descending_score(self):
        """Verify results are sorted by descending score."""
        raw_scores = [-2.0, 5.0, 1.0]
        reranker, _ = _make_reranker(raw_scores)
        results = await reranker.rerank("q", ["a", "b", "c"], top_k=None)
        assert results[0].text == "b"
        assert results[1].text == "c"
        assert results[2].text == "a"

    async def test_top_k_limits_results(self):
        """Verify top_k parameter limits the number of results."""
        raw_scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        reranker, _ = _make_reranker(raw_scores)
        results = await reranker.rerank("q", ["a", "b", "c", "d", "e"], top_k=2)
        assert len(results) == 2
        assert results[0].text == "e"
        assert results[1].text == "d"

    async def test_top_k_none_uses_constructor_default(self):
        """Verify None top_k uses constructor default."""
        reranker = SentenceTransformersReranker(top_k=3)
        reranker._model = _mock_cross_encoder([1.0, 2.0, 3.0, 4.0, 5.0])
        results = await reranker.rerank("q", ["a", "b", "c", "d", "e"])
        assert len(results) == 3

    async def test_rerank_result_fields(self):
        """Verify RerankResult has index, score, and text fields."""
        reranker, _ = _make_reranker([3.0])
        (result,) = await reranker.rerank("q", ["hello world"], top_k=None)
        assert isinstance(result, RerankResult)
        assert result.index == 0
        assert isinstance(result.score, float)
        assert result.text == "hello world"


class TestSentenceTransformersRerankerClose:
    """Tests for close / async context manager."""

    async def test_close_clears_model_reference(self):
        """Verify close clears the internal model reference."""
        reranker, _ = _make_reranker([1.0])
        assert reranker._model is not None
        await reranker.close()
        assert reranker._model is None

    async def test_close_is_safe_when_model_never_loaded(self):
        """Verify close is safe when model was never loaded."""
        reranker = SentenceTransformersReranker()
        await reranker.close()
        assert reranker._model is None

    async def test_async_context_manager_calls_close(self):
        """Verify async context manager calls close on exit."""
        reranker, _ = _make_reranker([1.0])
        async with reranker as ctx:
            assert ctx is reranker
        assert reranker._model is None
