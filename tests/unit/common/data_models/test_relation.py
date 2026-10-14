"""Tests for the ``Relation`` DataPoint."""

from __future__ import annotations

from am_diag.common.data_models import Relation


class TestRelationInstantiation:
    """Basic construction and field defaults."""

    def test_minimal_relation(self):
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
        rel = Relation(
            head_name="diabetes",
            tail_name="metformin",
            type="TREATED_BY",
            score=0.93,
            schema_properties={"treatmentLine": "first-line"},
        )
        assert rel.schema_properties == {"treatmentLine": "first-line"}


class TestRelationUUID:
    """Relation uses ``uuid4`` (no ``Dedup`` annotations)."""

    def test_relation_uses_uuid4(self):
        rel1 = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        rel2 = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        # Relations get different UUIDs since no Dedup annotation
        assert str(rel1.id) != str(rel2.id)

    def test_kind_is_relation(self):
        rel = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        assert rel.kind == "Relation"


class TestRelationProvenance:
    """Provenance tracking."""

    def test_provenance_chunk_ids_default_empty(self):
        rel = Relation(head_name="a", tail_name="b", type="TREATED_BY")
        assert rel.provenance.chunk_ids == []


class TestRelationSerialization:
    """Round-trip serialization."""

    def test_json_round_trip(self):
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
        rel = Relation(head_name="a", tail_name="b", type="X")
        d = rel.to_dict()
        restored = Relation.from_dict(d)
        assert str(restored.id) == str(rel.id)
        assert restored.type == "X"
