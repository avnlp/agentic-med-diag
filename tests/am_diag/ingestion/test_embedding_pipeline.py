"""Tests for am_diag/ingestion/embedding_pipeline.py."""

from __future__ import annotations

import pytest

from am_diag.common.data_models import Entity, Relation
from am_diag.common.data_models.provenance import Provenance
from am_diag.ingestion.embedding_pipeline import (
    build_embedding_pipeline,
    build_entity_records,
    build_relation_records,
)


pytestmark = pytest.mark.enable_socket


def _merged_entity(name: str, label: str, chunk_ids: list | None = None) -> Entity:
    return Entity(
        name=name,
        label=label,
        canonical_name=name,
        aliases=[name],
        score=0.95,
        provenance=Provenance(
            chunk_ids=chunk_ids or [],
            chunk_texts=["Metformin is a drug used for diabetes."],
        ),
    )


def _merged_relation(
    head: str,
    tail: str,
    rel_type: str,
) -> Relation:
    return Relation(
        head_name=head,
        tail_name=tail,
        type=rel_type,
        score=0.9,
        provenance=Provenance(
            chunk_ids=[],
            chunk_texts=["Metformin treats T2DM."],
        ),
    )


class TestBuildEntityRecords:
    def test_builds_records_from_entities(self):
        entities = [_merged_entity("metformin", "Drug")]
        chunk_text_map: dict = {}
        payloads, texts, ids = build_entity_records(entities, chunk_text_map)
        assert len(payloads) == 1
        assert "canonical_name" in payloads[0]
        assert "label" in payloads[0]
        assert payloads[0]["canonical_name"] == "metformin"
        assert payloads[0]["label"] == "Drug"

    def test_empty_list(self):
        payloads, texts, ids = build_entity_records([], {})
        assert payloads == []
        assert texts == []
        assert ids == []


class TestBuildRelationRecords:
    def test_builds_three_representations(self):
        relations = [_merged_relation("diabetes", "metformin", "TREATED_BY")]
        chunk_text_map: dict = {}
        payloads, texts, ids = build_relation_records(relations, chunk_text_map)
        assert len(payloads) == 3  # fwd, rev, typed
        assert "edge_text" in payloads[0]

    def test_empty_list(self):
        payloads, texts, ids = build_relation_records([], {})
        assert payloads == []
        assert texts == []
        assert ids == []


class TestBuildEmbeddingPipeline:
    def test_builds_pipeline(self):
        app = build_embedding_pipeline()
        assert app is not None
