"""Tests for the ``EntityType`` and ``RelationType`` DataPoints."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, RelationType


class TestEntityType:
    """Schema entity type definition DataPoint."""

    def test_minimal_entity_type(self):
        """Verify a minimally constructed EntityType has defaults."""
        et = EntityType(
            label="Drug",
            gliner_label="drug, medication, or pharmaceutical agent",
            description="A pharmaceutical drug or therapeutic agent.",
        )
        assert et.label == "Drug"
        assert et.gliner_label == "drug, medication, or pharmaceutical agent"
        assert et.key_property == "name"

    def test_explicit_metadata_preserved(self):
        """EntityType declares metadata explicitly, bypassing auto-derivation."""
        et = EntityType(
            label="Drug",
            gliner_label="drug",
            description="A drug",
        )
        assert "gliner_label" in et.metadata["index_fields"]
        assert "label" in et.metadata["identity_fields"]

    def test_uuid5_deterministic_via_label(self):
        """Verify same label produces same UUID."""
        """Same label → same UUID."""
        et1 = EntityType(label="Drug", gliner_label="drug", description="A")
        et2 = EntityType(label="Drug", gliner_label="drug", description="A")
        assert str(et1.id) == str(et2.id)

    def test_different_labels_different_uuids(self):
        """Verify different labels produce different UUIDs."""
        et1 = EntityType(label="Drug", gliner_label="drug", description="A")
        et2 = EntityType(label="Disease", gliner_label="disease", description="B")
        assert str(et1.id) != str(et2.id)

    def test_custom_key_property(self):
        """Verify custom key_property is stored."""
        et = EntityType(
            label="Gene",
            gliner_label="human gene or genetic locus",
            description="A human gene",
            key_property="symbol",
        )
        assert et.key_property == "symbol"

    def test_kind_is_entitytype(self):
        """Verify the kind discriminator is set to 'EntityType'."""
        et = EntityType(label="X", gliner_label="x", description="x")
        assert et.kind == "EntityType"


class TestRelationType:
    """Schema relation type definition DataPoint."""

    def test_minimal_relation_type(self):
        """Verify a minimally constructed RelationType has defaults."""
        rt = RelationType(type="TREATED_BY", glirel_label="treats")
        assert rt.type == "TREATED_BY"
        assert rt.glirel_label == "treats"
        assert rt.head_labels == []
        assert rt.tail_labels == []
        assert rt.properties == []

    def test_explicit_metadata_preserved(self):
        """Verify explicitly declared metadata is preserved."""
        rt = RelationType(type="TREATED_BY", glirel_label="treats")
        assert "glirel_label" in rt.metadata["index_fields"]
        assert "type" in rt.metadata["identity_fields"]

    def test_uuid5_deterministic_via_type(self):
        """Verify same type produces same UUID."""
        rt1 = RelationType(type="TREATED_BY", glirel_label="treats")
        rt2 = RelationType(type="TREATED_BY", glirel_label="treats")
        assert str(rt1.id) == str(rt2.id)

    def test_different_types_different_uuids(self):
        """Verify different types produce different UUIDs."""
        rt1 = RelationType(type="TREATED_BY", glirel_label="treats")
        rt2 = RelationType(type="CAUSES", glirel_label="causes")
        assert str(rt1.id) != str(rt2.id)

    def test_head_tail_labels(self):
        """Verify head and tail labels are stored."""
        rt = RelationType(
            type="TREATED_BY",
            glirel_label="treats",
            head_labels=["Disease"],
            tail_labels=["Drug", "Procedure"],
        )
        assert "Disease" in rt.head_labels
        assert "Drug" in rt.tail_labels
        assert "Procedure" in rt.tail_labels

    def test_custom_properties(self):
        """Verify custom properties are stored."""
        rt = RelationType(
            type="TREATED_BY",
            glirel_label="treats",
            properties=["treatmentLine", "evidenceLevel"],
        )
        assert "treatmentLine" in rt.properties
        assert "evidenceLevel" in rt.properties

    def test_kind_is_relationtype(self):
        """Verify the kind discriminator is set to 'RelationType'."""
        rt = RelationType(type="X", glirel_label="x")
        assert rt.kind == "RelationType"
