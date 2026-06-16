"""Data model tests for corpus loaders."""

from __future__ import annotations

from am_diag.common.data_models import Document


class TestDocument:
    def test_source_field(self):
        doc = Document(
            source="textbooks",
            text="text",
        )
        assert doc.source == "textbooks"

    def test_external_id_optional(self):
        doc = Document(
            source="pubmed",
            text="text",
            external_id="PMC123",
        )
        assert doc.external_id == "PMC123"

    def test_external_id_defaults_none(self):
        doc = Document(source="statpearls", text="text")
        assert doc.external_id is None

    def test_properties_defaults_to_empty_dict(self):
        doc = Document(source="statpearls", text="text")
        assert doc.properties == {}

    def test_properties_accepts_arbitrary_fields(self):
        doc = Document(
            source="pubmed",
            text="text",
            properties={"extra_field": "extra_value"},
        )
        assert doc.properties["extra_field"] == "extra_value"
