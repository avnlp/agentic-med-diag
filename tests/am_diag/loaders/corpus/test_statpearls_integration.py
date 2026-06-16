"""Integration tests for StatPearlsCorpusLoader."""

from __future__ import annotations

import asyncio

import pytest

from am_diag.loaders.corpus.statpearls import StatPearlsCorpusLoader


class TestStatPearlsCorpusLoaderIntegration:
    """Integration tests — requires network access."""

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_load_5_real_chunks(self):
        async def _run():
            return await StatPearlsCorpusLoader().load_sample(n=5)

        chunks = asyncio.run(_run())
        assert all(c.source == "statpearls" for c in chunks)
        assert all("article_id" in c.metadata for c in chunks)
        assert all("section_type" in c.metadata for c in chunks)
