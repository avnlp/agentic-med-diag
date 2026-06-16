"""Integration tests for PubmedCaseReportsCorpusLoader."""

from __future__ import annotations

import asyncio

import pytest

from am_diag.loaders.corpus.pubmed_case_reports import PubmedCaseReportsCorpusLoader


class TestPubmedCaseReportsCorpusLoaderIntegration:
    """Integration tests — requires network access."""

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_load_5_real_chunks(self):
        async def _run():
            return await PubmedCaseReportsCorpusLoader().load_sample(n=5)

        chunks = asyncio.run(_run())
        assert all(c.metadata["section_type"] == "full_text" for c in chunks)
        assert all(c.metadata["license"] == "CC-BY-4.0" for c in chunks)
