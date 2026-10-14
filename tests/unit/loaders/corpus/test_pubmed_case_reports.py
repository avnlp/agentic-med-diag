"""Unit tests for PubmedCaseReportsCorpusLoader."""

from __future__ import annotations

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.pubmed_case_reports import PubmedCaseReportsCorpusLoader


class TestPubmedCaseReportsCorpusLoader:
    """Order: Edge/empty → Core behavior → Fixtures."""

    def _make_row(self, **overrides):
        row = {
            "pmcid": "PMC123",
            "text": "Case report text.",
            "title": "Case Report",
            "journal": "NEJM",
            "publication_date": "2023-01-15",
        }
        row.update(overrides)
        return row

    def test_empty_text_returns_none(self):
        """Test that empty text returns None from row-to-document conversion."""
        loader = PubmedCaseReportsCorpusLoader()
        assert loader._row_to_document(self._make_row(text="")) is None

    def test_properties_has_all_required_keys(self):
        """Test that the document properties include all required metadata keys."""
        loader = PubmedCaseReportsCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert "pmcid" in doc.properties
        assert "journal" in doc.properties
        assert "publication_date" in doc.properties
        assert "section_type" in doc.properties
        assert "license" in doc.properties

    def test_valid_row_returns_document(self):
        """Test that a valid row produces a Document with the correct source."""
        loader = PubmedCaseReportsCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert isinstance(doc, Document)
        assert doc.source == "pubmed_case_reports"

    def test_pmcid_is_external_id(self):
        """Test that pmcid is used as the external_id of the Document."""
        loader = PubmedCaseReportsCorpusLoader()
        doc = loader._row_to_document(self._make_row(pmcid="PMC123"))
        assert doc.external_id == "PMC123"

    def test_section_type_is_full_text(self):
        """Test that section_type is set to 'full_text' by default."""
        loader = PubmedCaseReportsCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc.properties["section_type"] == "full_text"

    def test_license_is_cc_by_40(self):
        """Test that the license property is set to 'CC-BY-4.0'."""
        loader = PubmedCaseReportsCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc.properties["license"] == "CC-BY-4.0"

    def test_same_pmcid_produces_same_document_uuid(self):
        """Test that deterministic UUID generation for the same pmcid works."""
        loader = PubmedCaseReportsCorpusLoader()
        doc1 = loader._row_to_document(self._make_row(pmcid="PMC999"))
        doc2 = loader._row_to_document(self._make_row(pmcid="PMC999"))
        assert doc1.id == doc2.id

    def test_different_pmcids_produce_different_uuids(self):
        """Test that different pmcids produce distinct document UUIDs."""
        loader = PubmedCaseReportsCorpusLoader()
        doc1 = loader._row_to_document(self._make_row(pmcid="PMC001"))
        doc2 = loader._row_to_document(self._make_row(pmcid="PMC002"))
        assert doc1.id != doc2.id
