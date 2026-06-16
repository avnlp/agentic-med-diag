"""Unit tests for StatPearlsCorpusLoader."""

from __future__ import annotations

import pytest

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.statpearls import StatPearlsCorpusLoader


class TestStatPearlsCorpusLoaderRowToDocument:
    """Edge/empty → Core behavior → Fallback → Fixtures."""

    def _make_row(self, **overrides):
        row = {
            "id": "article-100024_2",
            "content": "Some text.",
            "title": "Chronic Total Occlusion -- Introduction",
        }
        row.update(overrides)
        return row

    def test_empty_content_returns_none(self):
        loader = StatPearlsCorpusLoader()
        assert loader._row_to_document(self._make_row(content="")) is None

    def test_valid_row_returns_document(self):
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
        loader = StatPearlsCorpusLoader()
        doc = loader._row_to_document(self._make_row(id=row_id))
        assert "article_id" in doc.properties
        assert doc.properties["article_id"] == expected_article
        assert "chunk_index" in doc.properties
        assert doc.properties["chunk_index"] == expected_idx

    @pytest.mark.parametrize(
        "title,expected_section",
        [
            ("Chronic Total Occlusion -- Introduction", "Introduction"),
            ("Pain Management -- Pharmacotherapy -- Details", "Details"),
            ("No Separator Title", "Unknown"),
        ],
    )
    def test_section_type_from_title(self, title, expected_section):
        loader = StatPearlsCorpusLoader()
        doc = loader._row_to_document(self._make_row(title=title))
        assert "section_type" in doc.properties
        assert doc.properties["section_type"] == expected_section
