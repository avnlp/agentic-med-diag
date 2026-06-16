"""Unit tests for ClinicalGuidelinesCorpusLoader astream behavior."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from datasets import Dataset

from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader


@pytest.mark.enable_socket
class TestGuidelinesAstream:
    """Tests for the base class astream via ClinicalGuidelinesCorpusLoader.

    Offset/limit now apply at the row level (standard base-class behavior)
    since each row maps to a single Document.
    """

    def test_limit_applies_to_rows(self, loader_with_mock):
        async def _run():
            chunks = []
            async for batch in loader_with_mock.astream(batch_size=100, limit=3):
                chunks.extend(batch)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 3

    def test_offset_skips_rows(self, loader_with_mock):
        async def _run():
            all_chunks = []
            async for batch in loader_with_mock.astream(batch_size=100):
                all_chunks.extend(batch)
            return all_chunks

        all_chunks = asyncio.run(_run())

        async def _run_offset():
            offset_chunks = []
            async for batch in loader_with_mock.astream(batch_size=100, offset=2):
                offset_chunks.extend(batch)
            return offset_chunks

        offset_chunks = asyncio.run(_run_offset())
        assert len(offset_chunks) == len(all_chunks) - 2

    def test_streams_all_rows(self, loader_with_mock):
        async def _run():
            chunks = []
            async for batch in loader_with_mock.astream(batch_size=100):
                chunks.extend(batch)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 5

    #  Fixtures

    @pytest.fixture
    def make_streaming_dataset(self):
        """Create a real IterableDataset from a list of dicts."""

        def _make(rows: list[dict[str, Any]]):
            return Dataset.from_list(rows).to_iterable_dataset()

        return _make

    @pytest.fixture
    def patch_guidelines_load_dataset(self, mocker, make_streaming_dataset):
        """Patch load_dataset with a real IterableDataset."""

        def _patch(module_path: str, rows: list[dict[str, Any]]):
            def _factory(*args: Any, **kwargs: Any):  # noqa: ARG001
                return make_streaming_dataset(rows)

            mocker.patch(f"{module_path}.load_dataset", side_effect=_factory)

        return _patch

    @pytest.fixture
    def loader_with_mock(self, patch_guidelines_load_dataset):
        rows = [
            {
                "id": f"g_{i}",
                "source": "WHO",
                "title": "T",
                "clean_text": "Clinical guideline text for testing.",
                "url": None,
                "overview": "",
            }
            for i in range(5)
        ]
        patch_guidelines_load_dataset("am_diag.loaders.corpus.base", rows)
        return ClinicalGuidelinesCorpusLoader()
