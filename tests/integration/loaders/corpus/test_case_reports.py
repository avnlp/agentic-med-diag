"""Integration tests for PubmedCaseReportsCorpusLoader.

These tests check for loading, source, ID format, and property invariants.
"""

from __future__ import annotations

import asyncio
import re

from am_diag.loaders.corpus.pubmed_case_reports import PubmedCaseReportsCorpusLoader


_PMCID_RE = re.compile(r"^PMC\d+$")


class TestPubmedCaseReportsCorpusLoader:
    """Tests for PubmedCaseReportsCorpusLoader — source, PMCID, and license."""

    # Allow ~10% empty-row drift margin so column renames won't silently lose rows.
    _LIMIT = 50

    def test_load_docs_non_empty_text(self):
        """Test that column renames are caught and don't silently drop rows."""
        chunks = asyncio.run(PubmedCaseReportsCorpusLoader().load_sample(n=self._LIMIT))
        assert len(chunks) >= 45, f"Expected ≥45 docs, got {len(chunks)}"
        assert all(c.text for c in chunks)

    def test_source_literal(self):
        """Test that the source-field literal hasn't changed in production."""
        chunks = asyncio.run(PubmedCaseReportsCorpusLoader().load_sample(n=self._LIMIT))
        assert all(c.source == "pubmed_case_reports" for c in chunks)

    def test_external_id_format(self):
        """Test that every external ID follows the ``PMC<digits>`` pattern."""
        chunks = asyncio.run(PubmedCaseReportsCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.external_id is not None, "external_id must not be None"
            assert _PMCID_RE.match(c.external_id)

    def test_derived_properties_non_empty(self):
        """Test that license, section_type, and journal are correctly populated."""
        chunks = asyncio.run(PubmedCaseReportsCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.properties["license"] == "CC-BY-4.0", (
                "license must be 'CC-BY-4.0' for PubMed Central case reports"
            )
            assert c.properties["section_type"] == "full_text"
            assert c.properties["journal"], "journal must be non-empty"
            assert c.properties["publication_date"], (
                "publication_date must be non-empty"
            )
