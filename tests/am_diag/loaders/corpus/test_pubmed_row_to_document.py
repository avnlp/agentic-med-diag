"""Unit tests for PubMedCorpusLoader."""

from __future__ import annotations

import pytest

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader


class TestPubMedCorpusLoaderRowToDocument:
    """Order: Edge/empty → Core behavior → Fixtures."""

    def _make_row(self, **overrides):
        row = {
            "id": "pubmed-12345678-0",
            "content": "Abstract text.",
            "title": "PubMed Article",
        }
        row.update(overrides)
        return row

    def test_empty_content_returns_none(self):
        loader = PubMedCorpusLoader()
        assert loader._row_to_document(self._make_row(content="")) is None

    def test_valid_row_returns_document(self):
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
        loader = PubMedCorpusLoader()
        doc = loader._row_to_document(self._make_row(id=row_id))
        assert "pubmed_id" in doc.properties
        assert doc.properties["pubmed_id"] == expected_pubmed_id
        assert "chunk_index" in doc.properties
        assert doc.properties["chunk_index"] == expected_idx
