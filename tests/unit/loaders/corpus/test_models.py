"""Tests for Document defaults, optional fields, and property behavior."""

from __future__ import annotations

from am_diag.common.data_models import Document


class TestDocument:
    """Tests for the Document data model."""

    def test_source_field(self):
        """Test that the source field is stored correctly."""
        doc = Document(
            source="textbooks",
            text="text",
        )
        assert doc.source == "textbooks"

    def test_external_id_optional(self):
        """Test that external_id can be set explicitly on a Document."""
        doc = Document(
            source="pubmed",
            text="text",
            external_id="PMC123",
        )
        assert doc.external_id == "PMC123"

    def test_external_id_defaults_none(self):
        """Test that external_id is None when not provided."""
        doc = Document(source="statpearls", text="text")
        assert doc.external_id is None

    def test_properties_defaults_to_empty_dict(self):
        """Test that properties defaults to an empty dict."""
        doc = Document(source="statpearls", text="text")
        assert doc.properties == {}

    def test_properties_accepts_arbitrary_fields(self):
        """Test that arbitrary key-value pairs can be stored in properties."""
        doc = Document(
            source="pubmed",
            text="text",
            properties={"extra_field": "extra_value"},
        )
        assert doc.properties["extra_field"] == "extra_value"
