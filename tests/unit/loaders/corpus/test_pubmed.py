"""_row_to_document behaviour for PubMed corpus rows.

Tests row-ID parsing (pubmed_id + chunk_index from ``pubmed-<id>-<N>``
format), empty-content skipping, source-field literal, and malformed-id
fallbacks (no chunk suffix, non-integer suffix).
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader


class TestPubMedCorpusLoader:
    """Tests for ``_row_to_document`` edge cases and ID fallbacks."""

    def _make_row(self, **overrides):
        row = {
            "id": "pubmed-12345678-0",
            "content": "Abstract text.",
            "title": "PubMed Article",
        }
        row.update(overrides)
        return row

    def test_empty_content_returns_none(self):
        """Test that empty content returns None from row-to-document."""
        loader = PubMedCorpusLoader()
        assert loader._row_to_document(self._make_row(content="")) is None

    def test_valid_row_returns_document(self):
        """Test that a valid row produces a Document with pubmed source."""
        loader = PubMedCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert isinstance(doc, Document)
        assert doc.source == "pubmed"

    @pytest.mark.parametrize(
        "row_id,expected_pubmed_id,expected_idx",
        [
            ("pubmed-12345678-0", "12345678", 0),
            ("pubmed-99999999-15", "99999999", 15),
            ("pubmed-1-0", "1", 0),
        ],
    )
    def test_id_parsing(self, row_id, expected_pubmed_id, expected_idx):
        """Test that pubmed_id and chunk_index are parsed from the row ID."""
        loader = PubMedCorpusLoader()
        doc = loader._row_to_document(self._make_row(id=row_id))
        assert "pubmed_id" in doc.properties
        assert doc.properties["pubmed_id"] == expected_pubmed_id
        assert "chunk_index" in doc.properties
        assert doc.properties["chunk_index"] == expected_idx

    def test_id_without_chunk_suffix_uses_whole_remainder(self):
        """Test that id like 'pubmed-12345678' (no suffix) uses stripped remainder."""
        loader = PubMedCorpusLoader()
        doc = loader._row_to_document(self._make_row(id="pubmed-12345678"))
        assert doc.properties["pubmed_id"] == "12345678"
        assert doc.properties["chunk_index"] == 0

    def test_non_integer_chunk_index_defaults_to_zero(self):
        """Test that a non-integer chunk suffix like 'pubmed-...-x' defaults to 0."""
        loader = PubMedCorpusLoader()
        doc = loader._row_to_document(self._make_row(id="pubmed-12345678-x"))
        assert doc.properties["pubmed_id"] == "12345678"
        assert doc.properties["chunk_index"] == 0
