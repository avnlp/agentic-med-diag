"""Tests for ``Provenance``."""

from __future__ import annotations

from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance


class TestProvenance:
    """Provenance field defaults and field behavior."""

    def test_extractor_defaults_to_none(self):
        prov = Provenance()
        assert prov.extractor is None

    def test_empty_collections_by_default(self):
        prov = Provenance()
        assert prov.sources == []
        assert prov.surface_forms == []
        assert prov.chunk_ids == []
        assert prov.chunk_texts == []

    def test_full_construction(self):
        prov = Provenance(
            extractor=ExtractorName.GLINER,
            sources=[ExtractorName.GLINER],
            surface_forms=["Metformin"],
            chunk_ids=[],
            chunk_texts=["metformin is a drug"],
            start_char=10,
            end_char=19,
        )
        assert prov.extractor == ExtractorName.GLINER
        assert prov.sources == [ExtractorName.GLINER]
        assert prov.surface_forms == ["Metformin"]
        assert prov.chunk_ids == []
        assert prov.chunk_texts == ["metformin is a drug"]
        assert prov.start_char == 10
        assert prov.end_char == 19

    def test_sources_list_mutation(self):
        prov = Provenance()
        prov.sources.append(ExtractorName.GLINER)
        prov.sources.append(ExtractorName.LLM)
        assert len(prov.sources) == 2

    def test_char_offset_none_by_default(self):
        prov = Provenance()
        assert prov.start_char is None
        assert prov.end_char is None
