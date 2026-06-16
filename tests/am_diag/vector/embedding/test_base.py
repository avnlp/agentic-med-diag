"""Tests for the Embedder ABC contract and async context manager behaviour."""

from __future__ import annotations

from abc import ABC

import pytest

from am_diag.vector.embedding.base import Embedder


class _FakeEmbedder(Embedder):
    """Minimal concrete subclass used to exercise the base-class behaviour."""

    def __init__(self) -> None:
        self.closed = False
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[float(len(t))] for t in texts]

    async def close(self) -> None:
        self.closed = True


class TestEmbedderIsAbstract:
    def test_is_abstract_base_class(self):
        assert issubclass(Embedder, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Embedder()  # type: ignore[abstract]

    def test_subclass_must_implement_embed_and_close(self):
        class _Incomplete(Embedder):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return []

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]


class TestEmbedderContract:
    async def test_embed_returns_list_of_lists_of_floats(self):
        embedder = _FakeEmbedder()
        result = await embedder.embed(["hello", "world!"])
        assert result == [[5.0], [6.0]]
        assert embedder.embed_calls == [["hello", "world!"]]

    async def test_embed_empty_input(self):
        embedder = _FakeEmbedder()
        assert await embedder.embed([]) == []


class TestEmbedderAsyncContextManager:
    async def test_aenter_returns_self(self):
        embedder = _FakeEmbedder()
        async with embedder as ctx:
            assert ctx is embedder

    async def test_aexit_calls_close(self):
        embedder = _FakeEmbedder()
        async with embedder:
            assert embedder.closed is False
        assert embedder.closed is True

    async def test_close_called_even_on_exception(self):
        embedder = _FakeEmbedder()
        with pytest.raises(RuntimeError):
            async with embedder:
                raise RuntimeError("boom")
        assert embedder.closed is True

    async def test_context_manager_usable_for_embedding(self):
        async with _FakeEmbedder() as embedder:
            result = await embedder.embed(["abc"])
        assert result == [[3.0]]
