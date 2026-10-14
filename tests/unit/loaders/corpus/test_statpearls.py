"""_row_to_document behaviour for StatPearls corpus rows.

Covers row-ID parsing (article_id + chunk_index from
``article-<id>_<N>`` format), empty-content skipping, source-field
literal, malformed-id fallbacks, and ``section_type`` extraction from
the title's ``-- <Section>`` suffix.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.statpearls import StatPearlsCorpusLoader


class TestStatPearlsCorpusLoader:
    """Tests for ``_row_to_document`` ID parsing and section extraction."""

    def _make_row(self, **overrides):
        row = {
            "id": "article-100024_2",
            "content": "Some text.",
            "title": "Chronic Total Occlusion -- Introduction",
        }
        row.update(overrides)
        return row

    def test_empty_content_returns_none(self):
        """Test that empty content returns None from row-to-document."""
        loader = StatPearlsCorpusLoader()
        assert loader._row_to_document(self._make_row(content="")) is None

    def test_valid_row_returns_document(self):
        """Test that a valid row produces a Document with statpearls source."""
        loader = StatPearlsCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert isinstance(doc, Document)
        assert doc.source == "statpearls"

    @pytest.mark.parametrize(
        "row_id,expected_article,expected_idx",
        [
            ("article-100024_2", "100024", 2),
            ("article-999_0", "999", 0),
            ("article-100024_10", "100024", 10),
        ],
    )
    def test_id_parsing(self, row_id, expected_article, expected_idx):
        """Test that article_id and chunk_index are parsed from the row ID."""
        loader = StatPearlsCorpusLoader()
        doc = loader._row_to_document(self._make_row(id=row_id))
        assert "article_id" in doc.properties
        assert doc.properties["article_id"] == expected_article
        assert "chunk_index" in doc.properties
        assert doc.properties["chunk_index"] == expected_idx

    def test_id_without_chunk_suffix_uses_whole_remainder(self):
        """Test that an id like 'article-Foo' (no suffix) uses the whole remainder."""
        loader = StatPearlsCorpusLoader()
        doc = loader._row_to_document(self._make_row(id="article-Foo"))
        assert doc.properties["article_id"] == "Foo"
        assert doc.properties["chunk_index"] == 0

    def test_non_integer_chunk_index_defaults_to_zero(self):
        """Test that a non-integer chunk suffix like 'article-Foo_bar' defaults to 0."""
        loader = StatPearlsCorpusLoader()
        doc = loader._row_to_document(self._make_row(id="article-Foo_bar"))
        assert doc.properties["article_id"] == "Foo"
        assert doc.properties["chunk_index"] == 0

    @pytest.mark.parametrize(
        "title,expected_section",
        [
            ("Chronic Total Occlusion -- Introduction", "Introduction"),
            ("Pain Management -- Pharmacotherapy -- Details", "Details"),
            ("No Separator Title", "Unknown"),
        ],
    )
    def test_section_type_from_title(self, title, expected_section):
        """Test that section_type is extracted correctly from the title field."""
        loader = StatPearlsCorpusLoader()
        doc = loader._row_to_document(self._make_row(title=title))
        assert "section_type" in doc.properties
        assert doc.properties["section_type"] == expected_section
