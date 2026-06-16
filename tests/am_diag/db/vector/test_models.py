"""Tests for vector store data models.

The old ``am_diag.db.vector.models`` module has been superseded by
``am_diag.common.data_models.vector_record`` and
``am_diag.common.data_models.enums``. This file tests those replacement models
and verifies the old module is no longer importable.
"""

from __future__ import annotations

import importlib
from uuid import UUID, uuid4

import pytest

from am_diag.common.data_models.enums import Distance
from am_diag.common.data_models.vector_record import VectorHit, VectorRecord


class TestVectorRecord:
    def test_defaults_empty_payload(self):
        p = VectorRecord(id=uuid4(), vector=[0.1, 0.2])
        assert p.payload == {}

    def test_stores_arbitrary_payload(self):
        p = VectorRecord(
            id=uuid4(),
            vector=[1.0],
            payload={"name": "BRCA1", "kind": "gene"},
        )
        assert p.payload["name"] == "BRCA1"

    def test_id_is_uuid(self):
        uid = uuid4()
        p = VectorRecord(id=uid, vector=[0.5])
        assert isinstance(p.id, UUID)
        assert p.id == uid


class TestVectorHit:
    def test_fields_accessible(self):
        uid = uuid4()
        r = VectorHit(id=uid, score=0.92, payload={"chunk_id": "c1"})
        assert r.id == uid
        assert r.score == 0.92
        assert r.payload["chunk_id"] == "c1"


class TestDistance:
    def test_all_variants_exist(self):
        assert Distance.COSINE
        assert Distance.EUCLID
        assert Distance.DOT

    def test_string_values(self):
        assert Distance.COSINE == "Cosine"
        assert Distance.EUCLID == "Euclid"
        assert Distance.DOT == "Dot"


class TestObsoleteModelsModule:
    def test_old_models_module_no_longer_importable(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("am_diag.db.vector.models")
