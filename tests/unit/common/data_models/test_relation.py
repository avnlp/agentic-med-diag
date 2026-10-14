"""Tests for the ``Relation`` DataPoint."""

from __future__ import annotations

from uuid import uuid4

from am_diag.common.data_models import Relation


class TestRelationInstantiation:
    """Basic construction and field defaults."""

    def test_minimal_relation(self):
        """Verify a minimally constructed relation has defaults."""
        rel = Relation(
            head_name="diabetes",
            tail_name="metformin",
            type="TREATED_BY",
        )
        assert rel.head_name == "diabetes"
        assert rel.tail_name == "metformin"
        assert rel.type == "TREATED_BY"
        assert rel.score == 1.0
        assert rel.head_id is None
        assert rel.tail_id is None

    def test_schema_properties(self):
        """Verify schema_properties is stored correctly."""
        rel = Relation(
            head_name="diabetes",
            tail_name="metformin",
            type="TREATED_BY",
            score=0.93,
            schema_properties={"treatmentLine": "first-line"},
        )
        assert rel.schema_properties == {"treatmentLine": "first-line"}


class TestRelationUUID:
    """Relation identity is a deterministic ``uuid5`` over its endpoint keys."""

    def test_same_endpoints_and_type_give_same_id(self):
        """Two relations with equal endpoint ids and type share one id."""
        head, tail = uuid4(), uuid4()
        rel1 = Relation(head_id=head, tail_id=tail, type="TREATED_BY")
        rel2 = Relation(head_id=head, tail_id=tail, type="TREATED_BY")
        assert rel1.id == rel2.id

    def test_differing_type_gives_different_id(self):
        """Changing the relation type changes the deterministic id."""
        head, tail = uuid4(), uuid4()
        rel1 = Relation(head_id=head, tail_id=tail, type="TREATED_BY")
        rel2 = Relation(head_id=head, tail_id=tail, type="CAUSES")
        assert rel1.id != rel2.id

    def test_differing_endpoints_give_different_ids(self):
        """Changing either endpoint id changes the deterministic id."""
        head, tail = uuid4(), uuid4()
        base = Relation(head_id=head, tail_id=tail, type="TREATED_BY")
        other_head = Relation(head_id=uuid4(), tail_id=tail, type="TREATED_BY")
        other_tail = Relation(head_id=head, tail_id=uuid4(), type="TREATED_BY")
        assert base.id != other_head.id
        assert base.id != other_tail.id

    def test_none_endpoints_fall_back_to_uuid4(self):
        """Without endpoint ids the id is a fresh uuid4 per construction."""
        rel1 = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        rel2 = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        assert rel1.head_id is None
        assert rel1.id != rel2.id

    def test_kind_is_relation(self):
        """Verify the kind discriminator is set to 'Relation'."""
        rel = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        assert rel.kind == "Relation"


class TestRelationProvenance:
    """Provenance tracking."""

    def test_provenance_defaults_to_none(self):
        """Provenance is optional and defaults to None.

        Text extraction sets it; structured-graph loaders leave it unset.
        """
        rel = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        assert rel.provenance is None


class TestRelationSerialization:
    """Round-trip serialization."""

    def test_json_round_trip(self):
        """Verify JSON serialization round-trip preserves Relation fields."""
        rel = Relation(
            head_name="diabetes",
            tail_name="metformin",
            type="TREATED_BY",
            score=0.93,
        )
        json_str = rel.to_json()
        restored = Relation.from_json(json_str)
        assert restored.head_name == "diabetes"
        assert restored.tail_name == "metformin"
        assert restored.type == "TREATED_BY"
        assert restored.score == 0.93

    def test_dict_round_trip(self):
        """Verify dict serialization round-trip preserves Relation fields."""
        rel = Relation(head_name="a", tail_name="b", type="X")
        d = rel.to_dict()
        restored = Relation.from_dict(d)
        assert str(restored.id) == str(rel.id)
        assert restored.type == "X"
