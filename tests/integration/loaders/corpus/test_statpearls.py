"""Integration tests for StatPearlsCorpusLoader.

These tests check for loading, source, ID format, and property invariants.
"""

from __future__ import annotations

import asyncio
import re

from am_diag.loaders.corpus.statpearls import StatPearlsCorpusLoader


_STATPEARLS_ID_RE = re.compile(r"^article-\w+_\d+$")


class TestStatPearlsCorpusLoader:
    """Tests for StatPearlsCorpusLoader — source, ID format, and section_type."""

    # Allow ~10% empty-row drift margin so column renames won't silently lose rows.
    _LIMIT = 50

    def test_load_docs_non_empty_text(self):
        """Test that column renames are caught and don't silently drop rows."""
        chunks = asyncio.run(StatPearlsCorpusLoader().load_sample(n=self._LIMIT))
        assert len(chunks) >= 45, f"Expected ≥45 docs, got {len(chunks)}"
        assert all(c.text for c in chunks)

    def test_source_literal(self):
        """Test that the source-field literal hasn't changed in production."""
        chunks = asyncio.run(StatPearlsCorpusLoader().load_sample(n=self._LIMIT))
        assert all(c.source == "statpearls" for c in chunks)

    def test_external_id_format(self):
        """Test that every external ID follows the ``article-<id>_<index>`` pattern."""
        chunks = asyncio.run(StatPearlsCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.external_id is not None, "external_id must not be None"
            assert _STATPEARLS_ID_RE.match(c.external_id)

    def test_derived_properties_non_empty(self):
        """Test that article_id, section_type are non-empty and chunk_index is valid."""
        chunks = asyncio.run(StatPearlsCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.properties["article_id"], "article_id must be non-empty"
            assert c.properties["section_type"], "section_type must be non-empty"
            assert "Unknown" not in c.properties["section_type"], (
                "section_type must not be 'Unknown' — title format may have changed"
            )
            assert isinstance(c.properties["chunk_index"], int)
            assert c.properties["chunk_index"] >= 0
