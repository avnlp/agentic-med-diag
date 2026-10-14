"""Integration tests for TextbooksCorpusLoader.

These tests check for loading, source, ID format, and property invariants.
"""

from __future__ import annotations

import asyncio
import re

from am_diag.loaders.corpus.textbooks import TextbooksCorpusLoader


_TEXTBOOKS_ID_RE = re.compile(r"^textbooks-.+_\d+$")


class TestTextbooksCorpusLoader:
    """Tests for TextbooksCorpusLoader — source, ID format, and book_name."""

    # Allow ~10% empty-row drift margin so column renames won't silently lose rows.
    _LIMIT = 50

    def test_load_docs_non_empty_text(self):
        """Test that column renames are caught and don't silently drop rows."""
        chunks = asyncio.run(TextbooksCorpusLoader().load_sample(n=self._LIMIT))
        assert len(chunks) >= 45, f"Expected ≥45 docs, got {len(chunks)}"
        assert all(c.text for c in chunks)

    def test_source_literal(self):
        """Test that the source-field literal hasn't changed in production."""
        chunks = asyncio.run(TextbooksCorpusLoader().load_sample(n=self._LIMIT))
        assert all(c.source == "textbooks" for c in chunks)

    def test_external_id_format(self):
        """Test that every external ID follows the actual dataset's ID pattern."""
        chunks = asyncio.run(TextbooksCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.external_id is not None, "external_id must not be None"
            assert _TEXTBOOKS_ID_RE.match(c.external_id)

    def test_derived_properties_non_empty(self):
        """Test that book_name is non-empty and chunk_index is valid."""
        chunks = asyncio.run(TextbooksCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.properties["book_name"], "book_name must be non-empty"
            assert isinstance(c.properties["chunk_index"], int)
            assert c.properties["chunk_index"] >= 0
