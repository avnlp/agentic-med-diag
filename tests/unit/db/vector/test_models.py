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
    """Tests for VectorRecord data model."""

    def test_defaults_empty_payload(self):
        """Verify payload defaults to empty dict."""
        p = VectorRecord(id=uuid4(), vector=[0.1, 0.2])
        assert p.payload == {}

    def test_stores_arbitrary_payload(self):
        """Verify arbitrary payload data is stored."""
        p = VectorRecord(
            id=uuid4(),
            vector=[1.0],
            payload={"name": "BRCA1", "kind": "gene"},
        )
        assert p.payload["name"] == "BRCA1"

    def test_id_is_uuid(self):
        """Verify VectorRecord stores the UUID identifier correctly."""
        uid = uuid4()
        p = VectorRecord(id=uid, vector=[0.5])
        assert isinstance(p.id, UUID)
        assert p.id == uid


class TestVectorHit:
    """Tests for VectorHit data model."""

    def test_fields_accessible(self):
        """Verify all fields are accessible."""
        uid = uuid4()
        r = VectorHit(id=uid, score=0.92, payload={"chunk_id": "c1"})
        assert r.id == uid
        assert r.score == 0.92
        assert r.payload["chunk_id"] == "c1"


class TestDistance:
    """Tests for Distance enum."""

    def test_all_variants_exist(self):
        """Verify all distance variants exist."""
        assert Distance.COSINE
        assert Distance.EUCLID
        assert Distance.DOT

    def test_string_values(self):
        """Verify distance enum has correct string values."""
        assert Distance.COSINE == "Cosine"
        assert Distance.EUCLID == "Euclid"
        assert Distance.DOT == "Dot"


class TestObsoleteModelsModule:
    """Tests that old module is no longer importable."""

    def test_old_models_module_no_longer_importable(self):
        """Verify old models module is no longer importable."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("am_diag.db.vector.models")
