"""Tests for the ``EntityType`` and ``RelationType`` DataPoints."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, RelationType


class TestEntityType:
    """Schema entity type definition DataPoint."""

    def test_minimal_entity_type(self):
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
        """Same label → same UUID."""
        et1 = EntityType(label="Drug", gliner_label="drug", description="A")
        et2 = EntityType(label="Drug", gliner_label="drug", description="A")
        assert str(et1.id) == str(et2.id)

    def test_different_labels_different_uuids(self):
        et1 = EntityType(label="Drug", gliner_label="drug", description="A")
        et2 = EntityType(label="Disease", gliner_label="disease", description="B")
        assert str(et1.id) != str(et2.id)

    def test_custom_key_property(self):
        et = EntityType(
            label="Gene",
            gliner_label="human gene or genetic locus",
            description="A human gene",
            key_property="symbol",
        )
        assert et.key_property == "symbol"

    def test_kind_is_entitytype(self):
        et = EntityType(label="X", gliner_label="x", description="x")
        assert et.kind == "EntityType"


class TestRelationType:
    """Schema relation type definition DataPoint."""

    def test_minimal_relation_type(self):
        rt = RelationType(type="TREATED_BY", glirel_label="treats")
        assert rt.type == "TREATED_BY"
        assert rt.glirel_label == "treats"
        assert rt.head_labels == []
        assert rt.tail_labels == []
        assert rt.properties == []

    def test_explicit_metadata_preserved(self):
        rt = RelationType(type="TREATED_BY", glirel_label="treats")
        assert "glirel_label" in rt.metadata["index_fields"]
        assert "type" in rt.metadata["identity_fields"]

    def test_uuid5_deterministic_via_type(self):
        rt1 = RelationType(type="TREATED_BY", glirel_label="treats")
        rt2 = RelationType(type="TREATED_BY", glirel_label="treats")
        assert str(rt1.id) == str(rt2.id)

    def test_different_types_different_uuids(self):
        rt1 = RelationType(type="TREATED_BY", glirel_label="treats")
        rt2 = RelationType(type="CAUSES", glirel_label="causes")
        assert str(rt1.id) != str(rt2.id)

    def test_head_tail_labels(self):
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
        rt = RelationType(
            type="TREATED_BY",
            glirel_label="treats",
            properties=["treatmentLine", "evidenceLevel"],
        )
        assert "treatmentLine" in rt.properties
        assert "evidenceLevel" in rt.properties

    def test_kind_is_relationtype(self):
        rt = RelationType(type="X", glirel_label="x")
        assert rt.kind == "RelationType"
