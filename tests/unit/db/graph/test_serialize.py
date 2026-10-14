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
    def test_uuid_fields_become_strings(self):
        entity = _make_entity()
        params = to_params(entity)
        assert isinstance(params["id"], str)
        # Verify it's a valid UUID string
        UUID(params["id"])

    def test_dict_fields_become_json_strings(self):
        entity = _make_entity()
        params = to_params(entity)
        # schema_properties is a dict and should be JSON-serialized
        assert isinstance(params["schema_properties"], str)
        parsed = json.loads(params["schema_properties"])
        assert isinstance(parsed, dict)

    def test_drop_excludes_fields(self):
        entity = _make_entity()
        params = to_params(entity, drop={"score"})
        assert "score" not in params

    def test_primitives_pass_through(self):
        entity = _make_entity()
        params = to_params(entity)
        assert params["name"] == "metformin"

    def test_list_of_uuids_serialized(self):
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
    def test_shape_has_id_and_props(self):
        doc = _make_doc()
        rec = document_record(doc)
        assert "id" in rec
        assert "props" in rec
        assert "id" not in rec["props"]

    def test_id_is_string(self):
        doc = _make_doc()
        rec = document_record(doc)
        assert isinstance(rec["id"], str)
        UUID(rec["id"])


class TestChunkRecord:
    def test_shape_has_id_and_props(self):
        chunk = _make_chunk()
        rec = chunk_record(chunk)
        assert "id" in rec
        assert "props" in rec
        assert "id" not in rec["props"]

    def test_id_is_string(self):
        chunk = _make_chunk()
        rec = chunk_record(chunk)
        UUID(rec["id"])


class TestEntityRecord:
    def test_shape_has_id_type_label_props(self):
        entity = _make_entity()
        rec = entity_record(entity)
        assert "id" in rec
        assert "type_label" in rec
        assert "props" in rec

    def test_type_label_matches_entity_label(self):
        entity = _make_entity(label="Disease")
        rec = entity_record(entity)
        assert rec["type_label"] == "Disease"

    def test_label_not_in_props(self):
        entity = _make_entity()
        rec = entity_record(entity)
        assert "label" not in rec["props"]

    def test_id_not_in_props(self):
        entity = _make_entity()
        rec = entity_record(entity)
        assert "id" not in rec["props"]

    def test_id_is_string(self):
        entity = _make_entity()
        rec = entity_record(entity)
        UUID(rec["id"])


class TestRelationRecord:
    def test_shape_has_id_head_tail_props(self):
        rel = Relation(
            head_name="a", tail_name="b", type="CAUSES", provenance=Provenance()
        )
        rec = relation_record(rel)
        assert "id" in rec
        assert "head_id" in rec
        assert "tail_id" in rec
        assert "props" in rec

    def test_unresolved_head_tail_are_none(self):
        rel = Relation(head_name="a", tail_name="b", type="X", provenance=Provenance())
        rec = relation_record(rel)
        assert rec["head_id"] is None
        assert rec["tail_id"] is None

    def test_resolved_head_tail_are_strings(self):
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
        rel = Relation(head_name="a", tail_name="b", type="X", provenance=Provenance())
        rec = relation_record(rel)
        assert "head_id" not in rec["props"]
        assert "tail_id" not in rec["props"]


class TestCommunityRecord:
    def test_shape_has_id_and_props(self):
        c = _make_community()
        rec = community_record(c)
        assert "id" in rec
        assert "props" in rec
        assert "id" not in rec["props"]


class TestReportRecord:
    def test_shape_has_id_and_props(self):
        r = _make_report()
        rec = report_record(r)
        assert "id" in rec
        assert "props" in rec

    def test_findings_serialized_as_json(self):
        r = _make_report()
        rec = report_record(r)
        # findings is a list[Finding] (BaseModel list) — serialized via json.dumps
        findings_raw = rec["props"]["findings"]
        assert isinstance(findings_raw, str)
        parsed = json.loads(findings_raw)
        assert isinstance(parsed, list)


class TestHasEntityRecords:
    def test_empty_entities_returns_empty(self):
        assert has_entity_records([]) == []

    def test_entity_with_no_chunks_produces_no_records(self):
        entity = _make_entity()
        assert has_entity_records([entity]) == []

    def test_one_record_per_chunk_id(self):
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
        chunk_id = uuid4()
        entity = Entity(
            name="x",
            label="Drug",
            provenance=Provenance(chunk_ids=[chunk_id]),
        )
        recs = has_entity_records([entity])
        UUID(recs[0]["entity_id"])


class TestInCommunityRecords:
    def test_empty_communities_returns_empty(self):
        assert in_community_records([]) == []

    def test_one_record_per_entity_id(self):
        e1, e2 = uuid4(), uuid4()
        c = _make_community(entity_ids=[e1, e2])
        recs = in_community_records([c])
        assert len(recs) == 2
        entity_ids = {r["entity_id"] for r in recs}
        assert str(e1) in entity_ids

    def test_community_id_is_string(self):
        entity_id = uuid4()
        c = _make_community(entity_ids=[entity_id])
        recs = in_community_records([c])
        UUID(recs[0]["community_id"])


class TestParentCommunityRecords:
    def test_community_without_parent_excluded(self):
        c = _make_community()
        assert parent_community_records([c]) == []

    def test_community_with_parent_included(self):
        parent_id = uuid4()
        c = _make_community(parent=parent_id)
        recs = parent_community_records([c])
        assert len(recs) == 1
        assert recs[0]["parent_id"] == str(parent_id)
        UUID(recs[0]["child_id"])


class TestHasReportRecords:
    def test_empty_returns_empty(self):
        assert has_report_records([]) == []

    def test_one_record_per_report(self):
        cid = uuid4()
        r = _make_report(community_id=cid)
        recs = has_report_records([r])
        assert len(recs) == 1
        assert recs[0]["community_id"] == str(cid)
        UUID(recs[0]["report_id"])
