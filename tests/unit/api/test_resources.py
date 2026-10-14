"""Tests for the curation sub-resources.

A fake graph store backs each resource directly so hydration, cursor pagination,
the semantic-only update payload, and cascade delete can be checked without Neo4j.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from am_diag.api.errors import SchemaError
from am_diag.api.resources import EntityResource, RelationResource


def _entity_row(name: str = "sepsis") -> dict:
    """Build a Neo4j-shaped entity row for hydration."""
    return {
        "id": str(uuid4()),
        "labels": ["Entity", "Disease"],
        "props": {"id": str(uuid4()), "name": name, "score": 1.0},
    }


class TestEntityResource:
    """Tests for EntityResource list/update/delete."""

    async def test_list_sets_next_cursor_when_full_page(self):
        """Verify next_cursor is the last id when a full page is returned."""
        rows = [_entity_row(f"e{i}") for i in range(3)]
        store = SimpleNamespace(list_entities=AsyncMock(return_value=rows))
        page = await EntityResource(store).list(limit=3)
        assert page.next_cursor == str(page.items[-1].id)
        assert len(page) == 3

    async def test_list_no_cursor_when_partial_page(self):
        """Verify next_cursor is None when fewer than limit rows return."""
        store = SimpleNamespace(list_entities=AsyncMock(return_value=[_entity_row()]))
        page = await EntityResource(store).list(limit=100)
        assert page.next_cursor is None

    async def test_update_sends_only_semantic_keys(self):
        """Verify update forwards only semantic fields, never structural keys."""
        captured = {}

        async def fake_update(entity_id, props):
            captured["id"] = entity_id
            captured["props"] = props
            return _entity_row("renamed")

        store = SimpleNamespace(update_entity=fake_update)
        await EntityResource(store).update("eid", canonical_name="Acetylsalicylic acid")
        assert captured["props"] == {"canonical_name": "Acetylsalicylic acid"}
        assert "id" not in captured["props"]
        assert "label" not in captured["props"]

    async def test_update_rejects_structural_attribute(self):
        """Verify attributes touching a structural key raise SchemaError."""
        store = SimpleNamespace(update_entity=AsyncMock())
        with pytest.raises(SchemaError):
            await EntityResource(store).update("eid", attributes={"label": "X"})

    async def test_delete_calls_store(self):
        """Verify delete delegates to the store's cascade delete."""
        store = SimpleNamespace(delete_entity=AsyncMock())
        await EntityResource(store).delete("eid")
        store.delete_entity.assert_awaited_once_with("eid")


class TestRelationResource:
    """Tests for RelationResource update payloads."""

    async def test_update_maps_relation_type(self):
        """Verify relation_type is forwarded as the 'type' property."""
        captured = {}

        async def fake_update(relation_id, props):
            captured["props"] = props
            return {
                "id": str(uuid4()),
                "props": {"id": str(uuid4()), "type": "TREATS"},
                "head_id": str(uuid4()),
                "tail_id": str(uuid4()),
            }

        store = SimpleNamespace(update_relation=fake_update)
        await RelationResource(store).update("rid", relation_type="TREATS")
        assert captured["props"] == {"type": "TREATS"}
