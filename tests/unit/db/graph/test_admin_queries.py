"""Tests for the admin/curation Cypher templates and their client bindings.

Verifies each ``admin/*.cypher`` file loads, and that the Neo4jClient curation
methods bind the expected parameters. The Neo4j driver is patched at its
construction boundary; no database is touched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import RoutingControl

from am_diag.common.cypher.loader import load
from am_diag.db.graph.neo4j import MAX_NEIGHBOR_DEPTH, Neo4jClient
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

    async def test_entity_neighbors_clamps_depth_and_binds_limit(self, client):
        """Verify entity_neighbors caps the hop count and binds the id and limit."""
        await client.entity_neighbors("e1", depth=50, limit=7)
        kwargs = client._driver.execute_query.call_args.kwargs
        assert kwargs["parameters_"] == {"id": "e1", "limit": 7}
        assert (
            f"*1..{MAX_NEIGHBOR_DEPTH}]"
            in client._driver.execute_query.call_args.args[0]
        )

    @pytest.mark.parametrize("key", ["id", "label", "labels"])
    @pytest.mark.parametrize("method", ["update_entity", "update_relation"])
    async def test_update_rejects_reserved_keys(self, client, method, key):
        """Verify updates refuse structural keys without querying the database."""
        with pytest.raises(ValueError, match="reserved"):
            await getattr(client, method)("e1", {key: "x", "name": "y"})
        client._driver.execute_query.assert_not_called()

    async def test_update_relation_binds_id_and_props(self, client):
        """Verify update_relation binds the id and the props patch."""
        await client.update_relation("e1", {"weight": 1})
        params = client._driver.execute_query.call_args.kwargs["parameters_"]
        assert params == {"id": "e1", "props": {"weight": 1}}

    @pytest.mark.parametrize(
        ("method", "args", "params"),
        [
            ("get_entity", ("a",), {"id": "a"}),
            ("entity_relations", ("a",), {"id": "a"}),
            ("entity_chunks", ("a",), {"id": "a"}),
            ("get_relation", ("a",), {"id": "a"}),
            ("get_chunk", ("a",), {"id": "a"}),
            ("get_community", ("a",), {"id": "a"}),
            ("list_chunks", (), {"cursor": None, "limit": 100}),
            ("list_communities", (), {"cursor": None, "limit": 100}),
        ],
    )
    async def test_read_methods_bind_params_and_use_read_routing(
        self, client, method, args, params
    ):
        """Verify read methods bind their params and route to readers."""
        await getattr(client, method)(*args)
        kwargs = client._driver.execute_query.call_args.kwargs
        assert kwargs["parameters_"] == params
        assert kwargs["routing_"] == RoutingControl.READ

    @pytest.mark.parametrize("method", ["delete_entity", "delete_relation"])
    async def test_delete_methods_bind_id_and_use_write_routing(self, client, method):
        """Verify deletes bind the id and route to writers."""
        await getattr(client, method)("a")
        kwargs = client._driver.execute_query.call_args.kwargs
        assert kwargs["parameters_"] == {"id": "a"}
        assert kwargs["routing_"] == RoutingControl.WRITE
