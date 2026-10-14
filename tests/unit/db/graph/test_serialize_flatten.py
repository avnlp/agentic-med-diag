"""Tests for queryable-property flattening in the Neo4j serializer.

``to_params`` spreads ``schema_properties`` scalars and primitive lists into
individual (queryable) Neo4j properties, keeps genuinely-nested values in a
residual JSON block, and rejects keys that would shadow a core model field.
``from_props`` is the inverse and must round-trip ``schema_properties``.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.db.graph.serialize import (
    entity_record,
    from_props,
    relation_record,
    to_params,
)


class TestSerializeFlatten:
    """Spreading, residual nesting, collision guarding, and round-tripping."""

    def test_scalar_property_spreads_to_top_level(self) -> None:
        """A scalar schema property becomes its own top-level param."""
        ent = Entity(name="brca1", label="Gene", schema_properties={"biotype": "pc"})
        params = to_params(ent)
        assert params["biotype"] == "pc"

    def test_primitive_list_property_stays_native(self) -> None:
        """A list of primitives is kept as a native (queryable) Neo4j list."""
        ent = Entity(
            name="brca1", label="Gene", schema_properties={"synonyms": ["a", "b"]}
        )
        params = to_params(ent)
        assert params["synonyms"] == ["a", "b"]

    def test_nested_property_goes_to_residual_json(self) -> None:
        """A nested value is JSON-encoded under the residual schema_properties key."""
        ent = Entity(
            name="brca1",
            label="Gene",
            schema_properties={"evidence": {"direct": ["X"]}},
        )
        params = to_params(ent)
        assert "evidence" not in params
        residual = json.loads(params["schema_properties"])
        assert residual == {"evidence": {"direct": ["X"]}}

    def test_list_containing_dict_goes_to_residual(self) -> None:
        """A list that contains dicts is treated as nested, not a native list."""
        ent = Entity(
            name="brca1", label="Gene", schema_properties={"refs": [{"id": 1}]}
        )
        params = to_params(ent)
        assert "refs" not in params
        assert json.loads(params["schema_properties"]) == {"refs": [{"id": 1}]}

    def test_empty_schema_properties_emits_no_residual(self) -> None:
        """Text-path entities (no extra properties) get no schema_properties param."""
        ent = Entity(name="brca1", label="Gene")
        params = to_params(ent)
        assert "schema_properties" not in params

    def test_collision_with_core_field_raises(self) -> None:
        """A property named like a core field is rejected, not silently shadowed."""
        ent = Entity(name="brca1", label="Gene", schema_properties={"label": "x"})
        with pytest.raises(ValueError, match="collides with a core"):
            to_params(ent)

    def test_from_props_round_trips_schema_properties(self) -> None:
        """from_props reassembles spread + residual back into schema_properties."""
        original = {"biotype": "pc", "synonyms": ["a", "b"], "evidence": {"d": [1]}}
        ent = Entity(
            name="brca1", label="Gene", curie="ENSG1", schema_properties=original
        )
        rebuilt = from_props(Entity, to_params(ent))
        assert rebuilt.schema_properties == original
        assert rebuilt.curie == "ENSG1"
        assert rebuilt.name == "brca1"
        assert rebuilt.label == "Gene"


class TestPromotedProvenanceLists:
    """chunk_ids/surface_forms/sources are promoted to top-level union-able lists."""

    def test_entity_record_promotes_provenance_lists(self) -> None:
        """entity_record lifts chunk_ids/surface_forms to top-level string lists."""
        cid = uuid4()
        ent = Entity(
            name="brca1",
            label="Gene",
            provenance=Provenance(chunk_ids=[cid], surface_forms=["BRCA1"]),
        )
        rec = entity_record(ent)
        assert rec["chunk_ids"] == [str(cid)]
        assert rec["surface_forms"] == ["BRCA1"]

    def test_entity_record_empty_when_no_provenance(self) -> None:
        """A structured entity without provenance promotes empty lists."""
        ent = Entity(name="brca1", label="Gene")
        rec = entity_record(ent)
        assert rec["chunk_ids"] == []
        assert rec["surface_forms"] == []

    def test_relation_record_promotes_chunk_ids_and_keeps_sources(self) -> None:
        """relation_record promotes chunk_ids and lifts sources out of props."""
        cid = uuid4()
        rel = Relation(
            head_id=uuid4(),
            tail_id=uuid4(),
            head_name="a",
            tail_name="b",
            type="TREATED_BY",
            sources=["OPEN_TARGETS"],
            provenance=Provenance(chunk_ids=[cid]),
        )
        rec = relation_record(rel)
        assert rec["chunk_ids"] == [str(cid)]
        assert rec["sources"] == ["OPEN_TARGETS"]
        assert "sources" not in rec["props"]

    def test_from_props_does_not_leak_promoted_lists(self) -> None:
        """Promoted lists reconstruct via provenance, not schema_properties."""
        cid = uuid4()
        ent = Entity(
            name="brca1",
            label="Gene",
            schema_properties={"biotype": "pc"},
            provenance=Provenance(chunk_ids=[cid], surface_forms=["BRCA1"]),
        )
        rec = entity_record(ent)
        # Simulate the node property map after the Cypher union sets the lists.
        readback = {
            "id": rec["id"],
            "label": rec["type_label"],
            **rec["props"],
            "chunk_ids": rec["chunk_ids"],
            "surface_forms": rec["surface_forms"],
        }
        rebuilt = from_props(Entity, readback)
        assert rebuilt.schema_properties == {"biotype": "pc"}
        assert "chunk_ids" not in rebuilt.schema_properties
        assert "surface_forms" not in rebuilt.schema_properties
        assert rebuilt.provenance is not None
        assert rebuilt.provenance.chunk_ids == [cid]
