"""Tests for the Reranker ABC contract and async context manager behaviour."""

from __future__ import annotations

from abc import ABC

import pytest

from am_diag.vector.reranking.base import Reranker, RerankResult


class _FakeReranker(Reranker):
    """Minimal concrete subclass used to exercise the base-class behaviour."""

    def __init__(self) -> None:
        self.closed = False
        self.rerank_calls: list[tuple[str, list[str]]] = []

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        self.rerank_calls.append((query, documents))
        return [
            RerankResult(index=i, score=1.0 - i * 0.1, text=documents[i])
            for i in range(len(documents))
        ]

    async def close(self) -> None:
        self.closed = True


class TestRerankerIsAbstract:
    """Verify Reranker BC enforces abstract methods."""

    def test_is_abstract_base_class(self):
        """Verify Reranker is an abstract base class."""
        assert issubclass(Reranker, ABC)

    def test_cannot_instantiate_directly(self):
        """Verify Reranker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Reranker()  # type: ignore[abstract]

    def test_subclass_must_implement_rerank_and_close(self):
        """Verify subclass must implement both rerank and close."""

        class _Incomplete(Reranker):
            async def rerank(
                self,
                query: str,
                documents: list[str],
                *,
                top_k: int | None = None,
            ) -> list[RerankResult]:
                return []

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]


class TestRerankerContract:
    """Tests for Reranker contract — rerank behaviour."""

    async def test_rerank_returns_list_of_rerank_results(self):
        """Verify rerank returns a list of RerankResult instances."""
        reranker = _FakeReranker()
        results = await reranker.rerank("q", ["doc a", "doc b"])
        assert len(results) == 2
        assert all(isinstance(r, RerankResult) for r in results)
        assert reranker.rerank_calls == [("q", ["doc a", "doc b"])]

    async def test_rerank_empty_documents(self):
        """Verify rerank with empty documents returns empty list."""
        reranker = _FakeReranker()
        assert await reranker.rerank("q", []) == []


class TestRerankerAsyncContextManager:
    """Tests for async context manager protocol."""

    async def test_aenter_returns_self(self):
        """Verify __aenter__ returns self."""
        reranker = _FakeReranker()
        async with reranker as ctx:
            assert ctx is reranker

    async def test_aexit_calls_close(self):
        """Verify __aexit__ calls close."""
        reranker = _FakeReranker()
        async with reranker:
            assert reranker.closed is False
        assert reranker.closed is True

    async def test_close_called_even_on_exception(self):
        """Verify close is called even when an exception occurs."""
        reranker = _FakeReranker()
        with pytest.raises(RuntimeError):
            async with reranker:
                raise RuntimeError("boom")
        assert reranker.closed is True

    async def test_context_manager_usable_for_reranking(self):
        """Verify context manager is usable for reranking calls."""
        async with _FakeReranker() as reranker:
            results = await reranker.rerank("q", ["doc"])
        assert len(results) == 1
        assert results[0].text == "doc"
