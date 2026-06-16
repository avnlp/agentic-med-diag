"""Integration tests for PubMedCorpusLoader."""

from __future__ import annotations

import asyncio

import pytest

from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader


class TestPubMedCorpusLoaderIntegration:
    """Integration tests — requires network access."""

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_load_5_real_chunks(self):
        async def _run():
            return await PubMedCorpusLoader().load_sample(n=5)

        chunks = asyncio.run(_run())
        assert len(chunks) == 5
        assert all(c.source == "pubmed" for c in chunks)
        assert all("pubmed_id" in c.metadata for c in chunks)
