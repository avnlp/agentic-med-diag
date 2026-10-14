"""Tests for the admin/curation Cypher templates and their client bindings.

Verifies each ``admin/*.cypher`` file loads, and that the Neo4jClient curation
methods bind the expected parameters. The Neo4j driver is patched at its
construction boundary; no database is touched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from am_diag.common.cypher.loader import load
from am_diag.db.graph.neo4j import Neo4jClient
from am_diag.db.graph.settings import Neo4jSettings


_DRIVER_TARGET = "am_diag.db.graph.neo4j.AsyncGraphDatabase.driver"

_ADMIN_FILES = [
    "admin/get_entity",
    "admin/list_entities",
    "admin/entity_neighbors",
    "admin/entity_relations",
    "admin/entity_chunks",
    "admin/update_entity",
    "admin/delete_entity",
    "admin/get_relation",
    "admin/list_relations",
    "admin/update_relation",
    "admin/delete_relation",
    "admin/get_chunk",
    "admin/list_chunks",
    "admin/get_community",
    "admin/list_communities",
    "admin/wipe_graph",
]


@pytest.fixture
def client() -> Neo4jClient:
    """Return a Neo4jClient with a patched driver returning empty results."""
    driver = MagicMock()
    driver.close = AsyncMock()
    driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
    with patch(_DRIVER_TARGET, return_value=driver):
        c = Neo4jClient(Neo4jSettings(uri="bolt://h", user="u", password="p"))
    c._driver = driver
    return c


class TestAdminCypherFiles:
    """Tests that every admin Cypher template is present and loadable."""

    @pytest.mark.parametrize("name", _ADMIN_FILES)
    def test_loads_non_empty(self, name):
        """Verify each admin Cypher file loads with non-empty content."""
        text = load(name)
        assert text.strip()


class TestCurationBindings:
    """Tests that curation methods bind the expected Cypher parameters."""

    async def test_list_entities_binds_filters(self, client):
        """Verify list_entities binds cursor, limit, and label."""
        await client.list_entities(cursor="c1", limit=50, label="Drug")
        params = client._driver.execute_query.call_args.kwargs["parameters_"]
        assert params == {"cursor": "c1", "limit": 50, "label": "Drug"}

    async def test_list_relations_maps_relation_type_to_type(self, client):
        """Verify list_relations binds the relation_type as the 'type' param."""
        await client.list_relations(relation_type="TREATS", limit=10)
        params = client._driver.execute_query.call_args.kwargs["parameters_"]
        assert params == {"cursor": None, "limit": 10, "type": "TREATS"}

    async def test_update_entity_binds_id_and_props(self, client):
        """Verify update_entity binds the id and the props patch."""
        await client.update_entity("e1", {"name": "x"})
        params = client._driver.execute_query.call_args.kwargs["parameters_"]
        assert params == {"id": "e1", "props": {"name": "x"}}

    async def test_entity_neighbors_binds_depth(self, client):
        """Verify entity_neighbors binds the id and depth."""
        await client.entity_neighbors("e1", depth=2)
        params = client._driver.execute_query.call_args.kwargs["parameters_"]
        assert params == {"id": "e1", "depth": 2}
