"""Tests for am_diag/db/graph/serialize.py."""

from __future__ import annotations

import json
from uuid import UUID, uuid4

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport, Finding
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.db.graph.serialize import (
    chunk_record,
    community_record,
    document_record,
    entity_record,
    has_entity_records,
    has_report_records,
    in_community_records,
    parent_community_records,
    relation_record,
    report_record,
    to_params,
)


def _make_entity(name: str = "metformin", label: str = "Drug") -> Entity:
    return Entity(name=name, label=label, provenance=Provenance())


def _make_doc() -> Document:
    return Document(text="Sample text.", source="pubmed", external_id="pmid1")


def _make_chunk(doc_id: UUID | None = None) -> Chunk:
    did = doc_id or uuid4()
    return Chunk(
        text="chunk text",
        document_id=did,
        document_source="pubmed",
        chunk_index=0,
        chunk_size=10,
        cut_type="sentence",
    )


def _make_community(
    entity_ids: list[UUID] | None = None,
    parent: UUID | None = None,
) -> Community:
    return Community(
        community_id="0:1",
        entity_ids=entity_ids or [],
        parent=parent,
    )


def _make_report(community_id: UUID | None = None) -> CommunityReport:
    return CommunityReport(
        community_id=community_id or uuid4(),
        findings=[Finding(summary="s", explanation="e")],
    )


class TestToParams:
    """Tests for to_params helper."""

    def test_uuid_fields_become_strings(self):
        """Verify UUID fields become strings."""
        entity = _make_entity()
        params = to_params(entity)
        assert isinstance(params["id"], str)
        # Verify it's a valid UUID string
        UUID(params["id"])

    def test_nested_properties_become_residual_json(self):
        """Nested schema_properties are JSON-encoded under a residual key.

        Scalar/primitive-list properties are spread to top-level (see
        test_serialize_flatten.py); only genuinely nested values remain a JSON
        string under ``schema_properties``.
        """
        entity = Entity(
            name="metformin",
            label="Drug",
            schema_properties={"evidence": {"direct": ["a"]}},
        )
        params = to_params(entity)
        assert isinstance(params["schema_properties"], str)
        parsed = json.loads(params["schema_properties"])
        assert parsed == {"evidence": {"direct": ["a"]}}

    def test_empty_schema_properties_omits_residual_key(self):
        """An entity with no extra properties produces no schema_properties param."""
        entity = _make_entity()
        params = to_params(entity)
        assert "schema_properties" not in params

    def test_drop_excludes_fields(self):
        """Verify drop parameter excludes specified fields."""
        entity = _make_entity()
        params = to_params(entity, drop={"score"})
        assert "score" not in params

    def test_primitives_pass_through(self):
        """Verify primitive fields pass through unchanged."""
        entity = _make_entity()
        params = to_params(entity)
        assert params["name"] == "metformin"

    def test_list_of_uuids_serialized(self):
        """Verify lists of UUIDs are serialized."""
        chunk_id = uuid4()
        entity = Entity(
            name="x",
            label="Drug",
            provenance=Provenance(chunk_ids=[chunk_id]),
        )
        params = to_params(entity)
        # provenance is a nested BaseModel — should become JSON string
        assert isinstance(params["provenance"], str)


class TestDocumentRecord:
    """Tests for document_record helper."""

    def test_shape_has_id_and_props(self):
        """Verify record has id and props fields."""
        doc = _make_doc()
        rec = document_record(doc)
        assert "id" in rec
        assert "props" in rec
        assert "id" not in rec["props"]

    def test_id_is_string(self):
        """Verify id is a string."""
        doc = _make_doc()
        rec = document_record(doc)
        assert isinstance(rec["id"], str)
        UUID(rec["id"])


class TestChunkRecord:
    """Tests for chunk_record helper."""

    def test_shape_has_id_and_props(self):
        """Verify record has id and props fields."""
        chunk = _make_chunk()
        rec = chunk_record(chunk)
        assert "id" in rec
        assert "props" in rec
        assert "id" not in rec["props"]

    def test_id_is_string(self):
        """Verify record id is a valid UUID string for Neo4j compatibility."""
        chunk = _make_chunk()
        rec = chunk_record(chunk)
        UUID(rec["id"])


class TestEntityRecord:
    """Tests for entity_record helper."""

    def test_shape_has_id_type_label_props(self):
        """Verify record has id, type_label, and props fields."""
        entity = _make_entity()
        rec = entity_record(entity)
        assert "id" in rec
        assert "type_label" in rec
        assert "props" in rec

    def test_type_label_matches_entity_label(self):
        """Verify type_label matches entity label."""
        entity = _make_entity(label="Disease")
        rec = entity_record(entity)
        assert rec["type_label"] == "Disease"

    def test_label_not_in_props(self):
        """Verify label is not duplicated in props."""
        entity = _make_entity()
        rec = entity_record(entity)
        assert "label" not in rec["props"]

    def test_id_not_in_props(self):
        """Verify the entity id is not duplicated inside the props dict."""
        entity = _make_entity()
        rec = entity_record(entity)
        assert "id" not in rec["props"]

    def test_id_is_string(self):
        """Verify entity id is a valid UUID string for Neo4j compatibility."""
        entity = _make_entity()
        rec = entity_record(entity)
        UUID(rec["id"])


class TestRelationRecord:
    """Tests for relation_record helper."""

    def test_shape_has_id_head_tail_props(self):
        """Verify record has id, head_id, tail_id, and props."""
        rel = Relation(
            head_name="a", tail_name="b", type="CAUSES", provenance=Provenance()
        )
        rec = relation_record(rel)
        assert "id" in rec
        assert "head_id" in rec
        assert "tail_id" in rec
        assert "props" in rec

    def test_unresolved_head_tail_are_none(self):
        """Verify unresolved head/tail ids are None."""
        rel = Relation(head_name="a", tail_name="b", type="X", provenance=Provenance())
        rec = relation_record(rel)
        assert rec["head_id"] is None
        assert rec["tail_id"] is None

    def test_resolved_head_tail_are_strings(self):
        """Verify resolved head/tail ids are strings."""
        h_id = uuid4()
        t_id = uuid4()
        rel = Relation(
            head_name="a",
            tail_name="b",
            type="X",
            head_id=h_id,
            tail_id=t_id,
            provenance=Provenance(),
        )
        rec = relation_record(rel)
        assert rec["head_id"] == str(h_id)
        assert rec["tail_id"] == str(t_id)

    def test_head_tail_not_in_props(self):
        """Verify head_id and tail_id are not duplicated in props."""
        rel = Relation(head_name="a", tail_name="b", type="X", provenance=Provenance())
        rec = relation_record(rel)
        assert "head_id" not in rec["props"]
        assert "tail_id" not in rec["props"]


class TestCommunityRecord:
    """Tests for community_record helper."""

    def test_shape_has_id_and_props(self):
        """Verify record has id and props fields."""
        c = _make_community()
        rec = community_record(c)
        assert "id" in rec
        assert "props" in rec
        assert "id" not in rec["props"]


class TestReportRecord:
    """Tests for report_record helper."""

    def test_shape_has_id_and_props(self):
        """Verify record has id and props fields."""
        r = _make_report()
        rec = report_record(r)
        assert "id" in rec
        assert "props" in rec

    def test_findings_serialized_as_json(self):
        """Verify findings are serialized as JSON strings."""
        r = _make_report()
        rec = report_record(r)
        # findings is a list[Finding] (BaseModel list) — serialized via json.dumps
        findings_raw = rec["props"]["findings"]
        assert isinstance(findings_raw, str)
        parsed = json.loads(findings_raw)
        assert isinstance(parsed, list)


class TestHasEntityRecords:
    """Tests for has_entity_records helper."""

    def test_empty_entities_returns_empty(self):
        """Verify that no input entities produce no has-entity link records."""
        assert has_entity_records([]) == []

    def test_entity_with_no_chunks_produces_no_records(self):
        """Verify an entity with no chunk provenance produces no link records."""
        entity = _make_entity()
        assert has_entity_records([entity]) == []

    def test_one_record_per_chunk_id(self):
        """Verify one record is produced per chunk id."""
        c1, c2 = uuid4(), uuid4()
        entity = Entity(
            name="x",
            label="Drug",
            provenance=Provenance(chunk_ids=[c1, c2]),
        )
        recs = has_entity_records([entity])
        assert len(recs) == 2
        chunk_ids = {r["chunk_id"] for r in recs}
        assert str(c1) in chunk_ids
        assert str(c2) in chunk_ids

    def test_entity_id_is_string(self):
        """Verify entity id is a string."""
        chunk_id = uuid4()
        entity = Entity(
            name="x",
            label="Drug",
            provenance=Provenance(chunk_ids=[chunk_id]),
        )
        recs = has_entity_records([entity])
        UUID(recs[0]["entity_id"])


class TestInCommunityRecords:
    """Tests for in_community_records helper."""

    def test_empty_communities_returns_empty(self):
        """Verify communities without entity members produce no link records."""
        assert in_community_records([]) == []

    def test_one_record_per_entity_id(self):
        """Verify one record per entity id."""
        e1, e2 = uuid4(), uuid4()
        c = _make_community(entity_ids=[e1, e2])
        recs = in_community_records([c])
        assert len(recs) == 2
        entity_ids = {r["entity_id"] for r in recs}
        assert str(e1) in entity_ids

    def test_community_id_is_string(self):
        """Verify community id is a string."""
        entity_id = uuid4()
        c = _make_community(entity_ids=[entity_id])
        recs = in_community_records([c])
        UUID(recs[0]["community_id"])


class TestParentCommunityRecords:
    """Tests for parent_community_records helper."""

    def test_community_without_parent_excluded(self):
        """Verify community without parent is excluded."""
        c = _make_community()
        assert parent_community_records([c]) == []

    def test_community_with_parent_included(self):
        """Verify community with parent is included."""
        parent_id = uuid4()
        c = _make_community(parent=parent_id)
        recs = parent_community_records([c])
        assert len(recs) == 1
        assert recs[0]["parent_id"] == str(parent_id)
        UUID(recs[0]["child_id"])


class TestHasReportRecords:
    """Tests for has_report_records helper."""

    def test_empty_returns_empty(self):
        """Verify that no reports produce no has-report link records."""
        assert has_report_records([]) == []

    def test_one_record_per_report(self):
        """Verify one record per report."""
        cid = uuid4()
        r = _make_report(community_id=cid)
        recs = has_report_records([r])
        assert len(recs) == 1
        assert recs[0]["community_id"] == str(cid)
        UUID(recs[0]["report_id"])
