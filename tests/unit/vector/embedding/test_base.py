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

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[float(len(t))] for t in texts]

    async def close(self) -> None:
        self.closed = True


class TestEmbedderIsAbstract:
    """Tests for embedder is abstract."""

    def test_is_abstract_base_class(self):
        """Is abstract base class."""
        assert issubclass(Embedder, ABC)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate directly."""
        with pytest.raises(TypeError):
            Embedder()  # type: ignore[abstract]

    def test_abstract_members_are_embed_close_and_model_id(self):
        """The contract a concrete embedder must implement to be instantiable."""
        assert Embedder.__abstractmethods__ == frozenset({"embed", "close", "model_id"})


class TestEmbedderContract:
    """Tests for embedder contract."""

    def test_model_id_returns_identifier(self):
        """The model_id property exposes the embedder's identity."""
        assert _FakeEmbedder().model_id == "fake-model"

    async def test_embed_returns_list_of_lists_of_floats(self):
        """Embed returns list of lists of floats."""
        embedder = _FakeEmbedder()
        result = await embedder.embed(["hello", "world!"])
        assert result == [[5.0], [6.0]]
        assert embedder.embed_calls == [["hello", "world!"]]

    async def test_embed_empty_input(self):
        """Embed empty input."""
        embedder = _FakeEmbedder()
        assert await embedder.embed([]) == []


class TestEmbedderAsyncContextManager:
    """Tests for embedder async context manager."""

    async def test_aenter_returns_self(self):
        """Aenter returns self."""
        embedder = _FakeEmbedder()
        async with embedder as ctx:
            assert ctx is embedder

    async def test_aexit_calls_close(self):
        """Aexit calls close."""
        embedder = _FakeEmbedder()
        async with embedder:
            assert embedder.closed is False
        assert embedder.closed is True

    async def test_close_called_even_on_exception(self):
        """Close called even on exception."""
        embedder = _FakeEmbedder()
        with pytest.raises(RuntimeError):
            async with embedder:
                raise RuntimeError("boom")
        assert embedder.closed is True

    async def test_context_manager_usable_for_embedding(self):
        """Context manager usable for embedding."""
        async with _FakeEmbedder() as embedder:
            result = await embedder.embed(["abc"])
        assert result == [[3.0]]
