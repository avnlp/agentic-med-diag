"""Integration tests for ClinicalGuidelinesCorpusLoader."""

from __future__ import annotations

import asyncio

import pytest

from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader


class TestClinicalGuidelinesCorpusLoaderIntegration:
    """Integration tests — requires network access."""

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_load_5_real_chunks(self):
        async def _run():
            return await ClinicalGuidelinesCorpusLoader().load_sample(n=5)

        chunks = asyncio.run(_run())
        assert len(chunks) == 5
        assert all(c.source == "clinical_guidelines" for c in chunks)
        assert all("issuing_org" in c.metadata for c in chunks)
        assert all(c.metadata.get("issuing_org") for c in chunks)
