"""Integration tests for TextbooksCorpusLoader."""

from __future__ import annotations

import asyncio

import pytest

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.textbooks import TextbooksCorpusLoader


class TestTextbooksCorpusLoaderIntegration:
    """Integration tests — requires network access."""

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_load_5_real_chunks(self):
        async def _run():
            loader = TextbooksCorpusLoader()
            return await loader.load_sample(n=5)

        chunks = asyncio.run(_run())
        assert len(chunks) == 5
        assert all(isinstance(c, Document) for c in chunks)
        assert all(c.source == "textbooks" for c in chunks)
        assert all("book_name" in c.metadata for c in chunks)
        assert all(c.metadata["book_name"] for c in chunks)
        assert all(c.text for c in chunks)
