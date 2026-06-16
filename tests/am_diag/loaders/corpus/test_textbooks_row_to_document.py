"""Unit tests for TextbooksCorpusLoader._row_to_document."""

from __future__ import annotations

import pytest

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.textbooks import TextbooksCorpusLoader


class TestTextbooksCorpusLoaderRowToDocument:
    """Unit tests for _row_to_document — no async, no network.

    Order: Edge/empty → Core behavior → Fixtures.
    """

    def _make_row(self, **overrides):
        row = {
            "id": "textbooks-Gray_Anatomy_5",
            "content": "Text here.",
            "title": "Gray's Anatomy",
        }
        row.update(overrides)
        return row

    def test_empty_content_returns_none(self):
        loader = TextbooksCorpusLoader()
        assert loader._row_to_document(self._make_row(content="")) is None
        assert loader._row_to_document(self._make_row(content=None)) is None

    def test_valid_row_returns_document(self):
        loader = TextbooksCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert isinstance(doc, Document)
        assert doc.source == "textbooks"

    def test_external_id_equals_row_id(self):
        loader = TextbooksCorpusLoader()
        doc = loader._row_to_document(self._make_row(id="textbooks-Gray_Anatomy_5"))
        assert doc.external_id == "textbooks-Gray_Anatomy_5"

    def test_same_row_id_produces_same_uuid(self):
        loader = TextbooksCorpusLoader()
        doc1 = loader._row_to_document(self._make_row(id="textbooks-Gray_Anatomy_5"))
        doc2 = loader._row_to_document(self._make_row(id="textbooks-Gray_Anatomy_5"))
        assert doc1.id == doc2.id

    def test_title_passed_through(self):
        loader = TextbooksCorpusLoader()
        doc = loader._row_to_document(self._make_row(title="My Book"))
        assert doc.title == "My Book"

    @pytest.mark.parametrize(
        "row_id,expected_book,expected_idx",
        [
            ("textbooks-Gray_Anatomy_0", "Gray_Anatomy", 0),
            ("textbooks-Gray_Anatomy_12", "Gray_Anatomy", 12),
            (
                "textbooks-Harrison_Internal_Medicine_100",
                "Harrison_Internal_Medicine",
                100,
            ),
            ("textbooks-Simple_0", "Simple", 0),
        ],
    )
    def test_book_name_and_chunk_index_parsed(
        self,
        row_id,
        expected_book,
        expected_idx,
    ):
        loader = TextbooksCorpusLoader()
        doc = loader._row_to_document(self._make_row(id=row_id))
        assert "book_name" in doc.properties
        assert doc.properties["book_name"] == expected_book
        assert "chunk_index" in doc.properties
        assert doc.properties["chunk_index"] == expected_idx
