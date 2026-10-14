"""_row_to_document behaviour for clinical guidelines corpus rows.

Covers the one-row-to-one-Document contract: field mapping (title, source,
external_id), property extraction (issuing_org, url, overview, section_type,
guideline_id), deterministic UUID generation, and empty-clean_text skipping.
"""

from __future__ import annotations

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader


class TestClinicalGuidelinesCorpusLoader:
    """Row-to-document conversion for the clinical guidelines corpus loader."""

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
        """Test that rows are skipped when clean_text is empty."""
        loader = ClinicalGuidelinesCorpusLoader()
        assert loader._row_to_document(self._make_row(clean_text="")) is None

    def test_valid_row_returns_document(self):
        """Test that a well-formed row produces a Document with correct source."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert isinstance(doc, Document)

    def test_external_id_is_row_id(self):
        """Test that the row id field is passed through as external_id."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.external_id == "guideline_001"

    def test_same_id_produces_same_uuid(self):
        """Test that identical row IDs produce identical document IDs."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc1 = loader._row_to_document(self._make_row())
        doc2 = loader._row_to_document(self._make_row())
        assert doc1.id == doc2.id

    def test_source_is_guidelines(self):
        """Test that the source field is set to the clinical_guidelines literal."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.source == "clinical_guidelines"

    def test_has_title(self):
        """Test that row title is passed through as the document title."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.title == "Hypertension Guidelines"

    def test_issuing_org_from_source_field(self):
        """Test that the row source field maps to properties.issuing_org."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row(source="FDA"))
        assert doc is not None
        assert doc.properties["issuing_org"] == "FDA"

    def test_url_in_properties(self):
        """Test that row url is stored in properties.url."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row(url="https://fda.gov"))
        assert doc is not None
        assert doc.properties["url"] == "https://fda.gov"

    def test_overview_truncated_to_500_chars(self):
        """Test that properties.overview is capped at 500 characters."""
        loader = ClinicalGuidelinesCorpusLoader()
        long_overview = "x" * 600
        doc = loader._row_to_document(self._make_row(overview=long_overview))
        assert doc is not None
        assert len(doc.properties["overview"]) <= 500

    def test_section_type_in_properties(self):
        """Test that properties.section_type defaults to 'body'."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.properties["section_type"] == "body"

    def test_guideline_id_in_properties(self):
        """Test that properties.guideline_id is set from the row id field."""
        loader = ClinicalGuidelinesCorpusLoader()
        doc = loader._row_to_document(self._make_row())
        assert doc is not None
        assert doc.properties["guideline_id"] == "guideline_001"
