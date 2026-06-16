"""Tests for the ``Entity`` DataPoint."""

from __future__ import annotations

from am_diag.common.data_models import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance


class TestEntityInstantiation:
    """Basic construction and field defaults."""

    def test_minimal_entity(self):
        entity = Entity(name="metformin", label="Drug")
        assert entity.name == "metformin"
        assert entity.label == "Drug"
        assert entity.score == 1.0
        assert entity.schema_properties == {}
        assert entity.canonical_name is None
        assert entity.aliases == []

    def test_full_entity(self):
        entity = Entity(
            name="metformin",
            label="Drug",
            score=0.97,
            schema_properties={"mechanism": "biguanide"},
            canonical_name="metformin",
            aliases=["Metformin", "metformin hydrochloride"],
        )
        assert entity.score == 0.97
        assert entity.canonical_name == "metformin"
        assert "Metformin" in entity.aliases


class TestEntityUUID5:
    """Deterministic UUID5 dedup."""

    def test_same_name_label_same_uuid(self):
        e1 = Entity(name="metformin", label="Drug")
        e2 = Entity(name="metformin", label="Drug")
        assert str(e1.id) == str(e2.id)

    def test_different_labels_different_uuids(self):
        e1 = Entity(name="influenza", label="Disease")
        e2 = Entity(name="influenza", label="Pathogen")
        assert str(e1.id) != str(e2.id)


class TestEntityMetadata:
    """Auto-derived metadata from annotations."""

    def test_has_embeddable_name(self):
        entity = Entity(name="metformin", label="Drug")
        assert "name" in entity.metadata["index_fields"]

    def test_has_identity_fields(self):
        entity = Entity(name="metformin", label="Drug")
        assert "name" in entity.metadata["identity_fields"]
        assert "label" in entity.metadata["identity_fields"]

    def test_kind_is_entity(self):
        entity = Entity(name="x", label="Drug")
        assert entity.kind == "Entity"


class TestEntityProvenance:
    """Provenance tracking."""

    def test_provenance_sources_accumulate(self):
        entity = Entity(
            name="metformin",
            label="Drug",
            provenance=Provenance(sources=[ExtractorName.GLINER]),
        )
        entity.provenance.sources.append(ExtractorName.LLM)
        assert ExtractorName.GLINER in entity.provenance.sources
        assert ExtractorName.LLM in entity.provenance.sources

    def test_provenance_chunk_ids_default_empty(self):
        entity = Entity(name="metformin", label="Drug")
        assert entity.provenance.chunk_ids == []


class TestEntitySerialization:
    """Round-trip serialization with Entity-specific fields."""

    def test_json_round_trip(self):
        entity = Entity(
            name="metformin",
            label="Drug",
            score=0.97,
            canonical_name="metformin",
            provenance=Provenance(
                extractor=ExtractorName.GLINER,
                chunk_ids=[],
            ),
        )
        json_str = entity.to_json()
        restored = Entity.from_json(json_str)
        assert restored.name == "metformin"
        assert restored.label == "Drug"
        assert restored.score == 0.97
        assert restored.canonical_name == "metformin"
        assert restored.provenance.extractor == ExtractorName.GLINER

    def test_dict_round_trip(self):
        entity = Entity(name="test", label="Drug")
        d = entity.to_dict()
        restored = Entity.from_dict(d)
        assert str(restored.id) == str(entity.id)
        assert restored.name == "test"
