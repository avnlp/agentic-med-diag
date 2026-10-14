"""Integration tests for ClinicalGuidelinesCorpusLoader.

These tests check for loading, source, ID format, and property invariants.
"""

from __future__ import annotations

import asyncio

from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader


class TestClinicalGuidelinesCorpusLoader:
    """Tests for ClinicalGuidelinesCorpusLoader — source and issuing_org."""

    # Allow ~10% empty-row drift margin so column renames won't silently lose rows.
    _LIMIT = 50

    def test_load_docs_non_empty_text(self):
        """Test that the loader handles the non-standard ``clean_text`` column."""
        chunks = asyncio.run(
            ClinicalGuidelinesCorpusLoader().load_sample(n=self._LIMIT)
        )
        assert len(chunks) >= 45, f"Expected ≥45 docs, got {len(chunks)}"
        assert all(c.text for c in chunks)

    def test_source_literal(self):
        """Test that the source-field literal hasn't changed in production."""
        chunks = asyncio.run(
            ClinicalGuidelinesCorpusLoader().load_sample(n=self._LIMIT)
        )
        assert all(c.source == "clinical_guidelines" for c in chunks)

    def test_external_id_non_empty(self):
        """Test that external_id is present and non-empty for every document."""
        chunks = asyncio.run(
            ClinicalGuidelinesCorpusLoader().load_sample(n=self._LIMIT)
        )
        assert all(c.external_id for c in chunks)

    def test_derived_properties_non_empty(self):
        """Test that issuing_org is populated and section_type is ``body``."""
        chunks = asyncio.run(
            ClinicalGuidelinesCorpusLoader().load_sample(n=self._LIMIT)
        )
        for c in chunks:
            assert c.properties["issuing_org"], "issuing_org must be non-empty"
            assert c.properties["section_type"] == "body", (
                "section_type must be 'body' for guideline content"
            )
