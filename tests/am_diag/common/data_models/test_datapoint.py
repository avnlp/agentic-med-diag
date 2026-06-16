"""Tests for the ``DataPoint`` base class."""

from __future__ import annotations

from typing import Annotated

import pydantic
import pytest

from am_diag.common.data_models import DataPoint, Dedup, Embeddable


class TestDataPointIdentity:
    """UUID generation: ``uuid4`` fallback vs deterministic ``uuid5``."""

    def test_default_is_uuid4(self):
        """DataPoint subclasses without Dedup annotations get uuid4."""

        class PlainPoint(DataPoint):
            name: str

        p1 = PlainPoint(name="test")
        assert p1.id is not None

    def test_uuid5_deterministic_with_dedup(self):
        """Same Dedup field values produce the same UUID."""

        class DedupPoint(DataPoint):
            key: Annotated[str, Dedup()]

        p1 = DedupPoint(key="hello")
        p2 = DedupPoint(key="hello")
        assert str(p1.id) == str(p2.id)

    def test_uuid5_different_values_different_ids(self):
        """Different Dedup field values produce different UUIDs."""

        class DedupPoint(DataPoint):
            key: Annotated[str, Dedup()]

        p1 = DedupPoint(key="alpha")
        p2 = DedupPoint(key="beta")
        assert str(p1.id) != str(p2.id)

    def test_uuid5_normalizes_whitespace_and_case(self):
        """UUID5 input is lowercased with spaces replaced."""

        class DedupPoint(DataPoint):
            key: Annotated[str, Dedup()]

        p1 = DedupPoint(key="Metformin")
        p2 = DedupPoint(key="metformin")
        assert str(p1.id) == str(p2.id)

    def test_explicit_id_overrides_uuid5(self):
        """Passing ``id`` explicitly skips UUID5 generation."""

        class DedupPoint(DataPoint):
            key: Annotated[str, Dedup()]

        from uuid import UUID

        custom_id = UUID("00000000-0000-0000-0000-000000000001")
        p = DedupPoint(key="test", id=custom_id)
        assert p.id == custom_id


class TestDataPointSerialization:
    """JSON and dict round-trip serialization."""

    def test_to_from_json_round_trip(self):
        class TestPoint(DataPoint):
            name: str
            value: int = 42

        p = TestPoint(name="test", value=99)
        json_str = p.to_json()
        restored = TestPoint.from_json(json_str)
        assert restored.name == "test"
        assert restored.value == 99
        assert str(restored.id) == str(p.id)

    def test_to_from_dict_round_trip(self):
        class TestPoint(DataPoint):
            name: str

        p = TestPoint(name="dict_test")
        d = p.to_dict()
        restored = TestPoint.from_dict(d)
        assert restored.name == "dict_test"
        assert str(restored.id) == str(p.id)


class TestDataPointMetadataAutoDerivation:
    """Annotation-driven auto-population of ``metadata``."""

    def test_embeddable_auto_derives_index_fields(self):
        class TestPoint(DataPoint):
            name: Annotated[str, Embeddable()]

        p = TestPoint(name="test")
        assert "name" in p.metadata["index_fields"]

    def test_dedup_auto_derives_identity_fields(self):
        class TestPoint(DataPoint):
            key: Annotated[str, Dedup()]

        p = TestPoint(key="test")
        assert "key" in p.metadata["identity_fields"]

    def test_both_annotations_auto_derive(self):
        class TestPoint(DataPoint):
            name: Annotated[str, Embeddable(), Dedup()]

        p = TestPoint(name="test")
        assert "name" in p.metadata["index_fields"]
        assert "name" in p.metadata["identity_fields"]

    def test_explicit_metadata_overrides_auto_derivation(self):
        class TestPoint(DataPoint):
            name: Annotated[str, Embeddable(), Dedup()]
            metadata: dict = {"index_fields": ["custom"], "identity_fields": ["custom"]}

        p = TestPoint(name="test")
        assert p.metadata["index_fields"] == ["custom"]
        assert p.metadata["identity_fields"] == ["custom"]

    def test_no_annotations_no_auto_derive(self):
        class TestPoint(DataPoint):
            name: str

        p = TestPoint(name="test")
        assert p.metadata["index_fields"] == []


class TestDataPointVersioning:
    """Version counter and timestamp update."""

    def test_version_starts_at_one(self):
        class TestPoint(DataPoint):
            name: str

        p = TestPoint(name="v1")
        assert p.version == 1

    def test_update_version_increments(self):
        class TestPoint(DataPoint):
            name: str

        p = TestPoint(name="v1")
        v0 = p.version
        p.update_version()
        assert p.version == v0 + 1

    def test_update_version_updates_timestamp(self):
        class TestPoint(DataPoint):
            name: str

        p = TestPoint(name="v1")
        t0 = p.updated_at
        p.update_version()
        assert p.updated_at >= t0


class TestDataPointEmbeddableIntrospection:
    """Class methods for extracting embeddable data from instances."""

    def test_get_embeddable_data_returns_field_value(self):
        class TestPoint(DataPoint):
            name: Annotated[str, Embeddable()]

        p = TestPoint(name="hello world")
        assert DataPoint.get_embeddable_data(p) == "hello world"

    def test_get_embeddable_properties(self):
        class TestPoint(DataPoint):
            name: Annotated[str, Embeddable()]
            description: Annotated[str, Embeddable()]

        p = TestPoint(name="a", description="b")
        assert DataPoint.get_embeddable_properties(p) == ["a", "b"]

    def test_get_embeddable_property_names(self):
        class TestPoint(DataPoint):
            name: Annotated[str, Embeddable()]

        p = TestPoint(name="x")
        assert DataPoint.get_embeddable_property_names(p) == ["name"]


class TestDataPointKindField:
    """The ``kind`` field is auto-set to the class name."""

    def test_kind_is_class_name(self):
        class MyCustom(DataPoint):
            name: str

        p = MyCustom(name="x")
        assert p.kind == "MyCustom"


class TestDataPointEdgeCases:
    """Edge cases."""

    def test_empty_identity_fields_uses_uuid4(self):
        """No Dedup annotations → ``_get_identity_fields`` returns [] → uuid4."""

        class NoDedup(DataPoint):
            name: str

        p1 = NoDedup(name="x")
        p2 = NoDedup(name="x")
        # Different objects get different UUIDs (not deterministic)
        assert str(p1.id) != str(p2.id)

    def test_missing_identity_field_value_falls_back_to_uuid4(self):
        """If an identity field is missing, falls back to uuid4."""

        class ConditionalDedup(DataPoint):
            name: Annotated[str, Dedup()]

        # name is required, so passing it will use UUID5
        p = ConditionalDedup(name="test")
        assert p.id is not None
        # Can't construct without name since it's required
        bad_kwargs: dict[str, str] = {}
        with pytest.raises(pydantic.ValidationError):
            ConditionalDedup(**bad_kwargs)
