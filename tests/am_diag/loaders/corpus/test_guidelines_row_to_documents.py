"""Unit tests for ClinicalGuidelinesCorpusLoader._row_to_document."""

from __future__ import annotations

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader


class TestClinicalGuidelinesCorpusLoaderRowToDocument:
    """Tests for _row_to_document (one-row-to-one-document method)."""

    def _make_row(self, **overrides):
        row = {
            "id": "guideline_001",
            "source": "WHO",
            "title": "Hypertension Guidelines",
            "clean_text": "Hypertension is a common condition.",
            "url": "https://who.int",
            "overview": "Overview text.",
        }
        row.update(overrides)
        return row

    def test_empty_clean_text_returns_none(self):
        loader = ClinicalGuidelinesCorpusLoader()
        assert loader._row_to_document(self._make_row(clean_text="")) is None

    def test_valid_row_returns_document(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert isinstance(doc, Document)

    def test_external_id_is_row_id(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.external_id == "guideline_001"

    def test_same_id_produces_same_uuid(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc1 = loader._row_to_document(self._make_row())
        doc2 = loader._row_to_document(self._make_row())
        assert doc1.id == doc2.id

    def test_source_is_guidelines(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.source == "clinical_guidelines"

    def test_has_title(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.title == "Hypertension Guidelines"

    def test_issuing_org_from_source_field(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row(source="FDA"))
        assert doc is not None
        assert doc.properties["issuing_org"] == "FDA"

    def test_url_in_properties(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row(url="https://fda.gov"))
        assert doc is not None
        assert doc.properties["url"] == "https://fda.gov"

    def test_overview_truncated_to_500_chars(self):
        loader = ClinicalGuidelinesCorpusLoader()
        long_overview = "x" * 600
        doc = loader._row_to_document(self._make_row(overview=long_overview))
        assert doc is not None
        assert len(doc.properties["overview"]) <= 500

    def test_section_type_in_properties(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.properties["section_type"] == "body"

    def test_guideline_id_in_properties(self):
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.properties["guideline_id"] == "guideline_001"
