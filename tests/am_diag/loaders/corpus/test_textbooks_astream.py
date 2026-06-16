"""Unit tests for TextbooksCorpusLoader.astream()."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from datasets import Dataset

from am_diag.loaders.corpus.textbooks import TextbooksCorpusLoader


@pytest.mark.enable_socket
class TestTextbooksCorpusLoaderAstream:
    """Async unit tests for astream() — uses real IterableDataset, no network.

    Uses ``asyncio.run()`` instead of ``async def`` to avoid
    ``pytest-asyncio`` event-loop socket issues with ``pytest-socket``.

    Order: Core behavior → Fixtures.
    """

    def test_astream_yields_correct_batch_size(self, loader_and_ds):
        async def _run():
            batches = []
            async for batch in loader_and_ds.astream(batch_size=3):
                batches.append(batch)
            return batches

        batches = asyncio.run(_run())
        assert len(batches) >= 3
        assert all(len(b) <= 3 for b in batches)

    def test_astream_limit_stops_early(self, loader_and_ds):
        async def _run():
            chunks = []
            async for batch in loader_and_ds.astream(batch_size=10, limit=4):
                chunks.extend(batch)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 4

    def test_load_sample_returns_n_chunks(self, loader_and_ds):
        async def _run():
            return await loader_and_ds.load_sample(n=5)

        chunks = asyncio.run(_run())
        assert len(chunks) == 5

    def test_all_chunks_have_textbooks_source(self, loader_and_ds):
        async def _run():
            return await loader_and_ds.load_sample(n=10)

        chunks = asyncio.run(_run())
        assert all(c.source == "textbooks" for c in chunks)

    #  Fixtures (last, per Haystack convention)

    @pytest.fixture
    def make_streaming_dataset(self):
        def _make(rows: list[dict[str, Any]]):
            return Dataset.from_list(rows).to_iterable_dataset()

        return _make

    @pytest.fixture
    def patch_corpus_load_dataset(self, mocker, make_streaming_dataset):
        def _patch(module_path: str, rows: list[dict[str, Any]]):
            def _factory(*args: Any, **kwargs: Any):  # noqa: ARG001
                return make_streaming_dataset(rows)

            mocker.patch(f"{module_path}.load_dataset", side_effect=_factory)

        return _patch

    @pytest.fixture
    def loader_and_ds(self, patch_corpus_load_dataset):
        rows = [
            {"id": f"textbooks-Book_{i}", "content": f"text {i}", "title": "Book"}
            for i in range(10)
        ]
        patch_corpus_load_dataset("am_diag.loaders.corpus.base", rows)
        return TextbooksCorpusLoader()
