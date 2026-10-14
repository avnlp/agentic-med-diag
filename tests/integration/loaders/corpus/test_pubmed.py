"""Integration tests for PubMedCorpusLoader.

These tests check for loading, source, ID format, and property invariants.
"""

from __future__ import annotations

import asyncio
import re

from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader


_PUBMED_ID_RE = re.compile(r"^pubmed\d+n\d+_\d+$")


class TestPubMedCorpusLoader:
    """Tests for PubMedCorpusLoader — source, ID format, and property invariants."""

    # Allow ~10% empty-row drift margin so column renames won't silently lose rows.
    _LIMIT = 50

    def test_load_docs_non_empty_text(self):
        """Test that column renames are caught and don't silently drop rows."""
        chunks = asyncio.run(PubMedCorpusLoader().load_sample(n=self._LIMIT))
        assert len(chunks) >= 45, f"Expected ≥45 docs, got {len(chunks)}"
        assert all(c.text for c in chunks)

    def test_source_literal(self):
        """Test that the source-field literal hasn't changed in production."""
        chunks = asyncio.run(PubMedCorpusLoader().load_sample(n=self._LIMIT))
        assert all(c.source == "pubmed" for c in chunks)

    def test_external_id_format(self):
        """Test that every external ID follows the dataset's actual ID pattern."""
        chunks = asyncio.run(PubMedCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.external_id is not None, "external_id must not be None"
            assert _PUBMED_ID_RE.match(c.external_id)
            source_id, _, chunk_index = c.external_id.rpartition("_")
            assert c.properties["pubmed_id"] == source_id
            assert c.properties["chunk_index"] == int(chunk_index)

    def test_derived_properties_non_empty(self):
        """Test that pubmed_id and chunk_index are present with valid types."""
        chunks = asyncio.run(PubMedCorpusLoader().load_sample(n=self._LIMIT))
        for c in chunks:
            assert c.properties["pubmed_id"], "pubmed_id must be non-empty"
            assert isinstance(c.properties["chunk_index"], int)
            assert c.properties["chunk_index"] >= 0
