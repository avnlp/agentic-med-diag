"""Tests formerly covering entity_relation_queries.py.

That module has been superseded by am_diag/db/graph/serialize.py and the
.cypher files in am_diag/common/cypher/. Full coverage of the replacement is
in test_serialize.py. This file retains a minimal smoke-test to confirm the
old symbols are gone and the new ones are importable.
"""

from __future__ import annotations

import importlib

import pytest


class TestObsoleteModuleRemoved:
    """Verify obsolete modules are no longer importable."""

    def test_entity_relation_queries_no_longer_importable(self):
        """Verify entity_relation_queries module is removed."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("am_diag.db.graph.entity_relation_queries")

    def test_community_queries_no_longer_importable(self):
        """Verify community_queries module is removed."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("am_diag.db.graph.community_queries")


class TestSerializeModuleImportable:
    """Verify new serialize module is importable."""

    def test_serialize_importable(self):
        """Verify serialize module can be imported."""
        from am_diag.db.graph import serialize

        assert serialize is not None

    def test_key_functions_importable(self):
        """Verify key functions are importable from serialize module."""
        from am_diag.db.graph.serialize import (  # noqa: PLC0415
            entity_record,
            has_entity_records,
            relation_record,
        )

        assert callable(entity_record)
        assert callable(relation_record)
        assert callable(has_entity_records)
