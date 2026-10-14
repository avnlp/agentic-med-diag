"""Curation and inspection sub-resources for a Graph.

``graph.entities`` / ``graph.relations`` expose get / list (cursor-paginated) /
neighbors / relations / chunks / update / delete; ``graph.chunks`` and
``graph.communities`` are read-only. Updates carry only semantic fields —
structural identity (ids, labels, endpoints) is immutable. Entity deletes cascade
to incident relations; relation deletes are leaf operations.

Rows returned by the store are hydrated back into domain models via
``db.graph.serialize.from_props``.
"""

from __future__ import annotations

from typing import Any

from am_diag.api.errors import (
    EntityNotFoundError,
    RelationNotFoundError,
    SchemaError,
)
from am_diag.api.pagination import Page
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.db.graph.base import GraphStore
from am_diag.db.graph.serialize import from_props


# The resources expose a ``.list()`` method that shadows the builtin ``list``
# inside the class namespace; this alias keeps ``list[...]`` annotations valid.
_List = list

_IMMUTABLE_KEYS = frozenset({"id", "label", "kind", "head_id", "tail_id"})


def _page(items: _List[Any], limit: int) -> Page[Any]:
    """Wrap items in a page; the next cursor is the last id of a full page."""
    next_cursor = str(items[-1].id) if len(items) == limit else None
    return Page(items, next_cursor)


def _entity_from_row(row: dict[str, Any]) -> Entity:
    """Hydrate an Entity from an ``{id, labels, props}`` row.

    The secondary Neo4j label is reattached because ``label`` is stored as a node
    label, not a property.
    """
    props = dict(row.get("props") or {})
    if "label" not in props:
        secondary = [lbl for lbl in row.get("labels", []) if lbl != "Entity"]
        props["label"] = secondary[0] if secondary else "Entity"
    return from_props(Entity, props)


def _relation_from_row(row: dict[str, Any]) -> Relation:
    """Hydrate a Relation from an ``{id, props, head_id, tail_id}`` row."""
    props = dict(row.get("props") or {})
    props["head_id"] = row.get("head_id")
    props["tail_id"] = row.get("tail_id")
    return from_props(Relation, props)


def _chunk_from_row(row: dict[str, Any]) -> Chunk:
    """Hydrate a Chunk from an ``{id, props}`` row."""
    return from_props(Chunk, dict(row.get("props") or {}))


def _community_from_row(row: dict[str, Any]) -> Community:
    """Hydrate a Community from an ``{id, props}`` row."""
    return from_props(Community, dict(row.get("props") or {}))


def _semantic_patch(
    fields: dict[str, Any], attributes: dict[str, Any] | None
) -> dict[str, Any]:
    """Build a property patch from explicit semantic fields plus attributes.

    Args:
        fields: Named semantic fields with ``None`` values dropped.
        attributes: Optional free-form schema properties (primitive values).

    Returns:
        The merged property map.

    Raises:
        SchemaError: If an attribute key collides with a structural identity key.
    """
    patch = {k: v for k, v in fields.items() if v is not None}
    for key, value in (attributes or {}).items():
        if key in _IMMUTABLE_KEYS:
            raise SchemaError(f"Cannot modify structural key: {key!r}")
        patch[key] = value
    return patch


class EntityResource:
    """Curation resource for entities (``graph.entities``)."""

    def __init__(self, graph_store: GraphStore) -> None:
        """Bind the resource to its owning graph's store."""
        self._store = graph_store

    async def get(self, entity_id: str) -> Entity:
        """Return one entity.

        Raises:
            EntityNotFoundError: If no entity has that id.
        """
        row = await self._store.get_entity(entity_id)
        if row is None:
            raise EntityNotFoundError(f"No such entity: {entity_id!r}")
        return _entity_from_row(row)

    async def list(
        self, *, cursor: str | None = None, limit: int = 100, label: str | None = None
    ) -> Page[Entity]:
        """Return a cursor-paginated page of entities ordered by id."""
        rows = await self._store.list_entities(cursor=cursor, limit=limit, label=label)
        items = [_entity_from_row(r) for r in rows]
        return _page(items, limit)

    async def neighbors(self, entity_id: str, *, depth: int = 1) -> _List[Entity]:
        """Return entities within ``depth`` relation hops of ``entity_id``."""
        rows = await self._store.entity_neighbors(entity_id, depth)
        return [_entity_from_row(r) for r in rows]

    async def relations(self, entity_id: str) -> _List[Relation]:
        """Return relations incident to ``entity_id``."""
        rows = await self._store.entity_relations(entity_id)
        return [_relation_from_row(r) for r in rows]

    async def chunks(self, entity_id: str) -> _List[Chunk]:
        """Return chunks that mention ``entity_id``."""
        rows = await self._store.entity_chunks(entity_id)
        return [_chunk_from_row(r) for r in rows]

    async def update(
        self,
        entity_id: str,
        *,
        name: str | None = None,
        canonical_name: str | None = None,
        aliases: _List[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Entity:
        """Patch an entity's semantic fields (structural identity is immutable).

        Raises:
            EntityNotFoundError: If no entity has that id.
            SchemaError: If ``attributes`` touches a structural key.
        """
        patch = _semantic_patch(
            {"name": name, "canonical_name": canonical_name, "aliases": aliases},
            attributes,
        )
        row = await self._store.update_entity(entity_id, patch)
        if row is None:
            raise EntityNotFoundError(f"No such entity: {entity_id!r}")
        return _entity_from_row(row)

    async def delete(self, entity_id: str) -> None:
        """Delete an entity, cascading to all incident relations."""
        await self._store.delete_entity(entity_id)


class RelationResource:
    """Curation resource for relations (``graph.relations``)."""

    def __init__(self, graph_store: GraphStore) -> None:
        """Bind the resource to its owning graph's store."""
        self._store = graph_store

    async def get(self, relation_id: str) -> Relation:
        """Return one relation.

        Raises:
            RelationNotFoundError: If no relation has that id.
        """
        row = await self._store.get_relation(relation_id)
        if row is None:
            raise RelationNotFoundError(f"No such relation: {relation_id!r}")
        return _relation_from_row(row)

    async def list(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
        relation_type: str | None = None,
    ) -> Page[Relation]:
        """Return a cursor-paginated page of relations ordered by id."""
        rows = await self._store.list_relations(
            cursor=cursor, limit=limit, relation_type=relation_type
        )
        items = [_relation_from_row(r) for r in rows]
        return _page(items, limit)

    async def update(
        self,
        relation_id: str,
        *,
        relation_type: str | None = None,
        description: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Relation:
        """Patch a relation's semantic fields (endpoints are immutable).

        Raises:
            RelationNotFoundError: If no relation has that id.
            SchemaError: If ``attributes`` touches a structural key.
        """
        patch = _semantic_patch(
            {"type": relation_type, "description": description}, attributes
        )
        row = await self._store.update_relation(relation_id, patch)
        if row is None:
            raise RelationNotFoundError(f"No such relation: {relation_id!r}")
        return _relation_from_row(row)

    async def delete(self, relation_id: str) -> None:
        """Delete a single relation (endpoints are untouched)."""
        await self._store.delete_relation(relation_id)


class ChunkResource:
    """Read-only resource for chunks (``graph.chunks``)."""

    def __init__(self, graph_store: GraphStore) -> None:
        """Bind the resource to its owning graph's store."""
        self._store = graph_store

    async def get(self, chunk_id: str) -> Chunk | None:
        """Return one chunk, or ``None`` if absent."""
        row = await self._store.get_chunk(chunk_id)
        return _chunk_from_row(row) if row else None

    async def list(self, *, cursor: str | None = None, limit: int = 100) -> Page[Chunk]:
        """Return a cursor-paginated page of chunks ordered by id."""
        rows = await self._store.list_chunks(cursor=cursor, limit=limit)
        items = [_chunk_from_row(r) for r in rows]
        return _page(items, limit)


class CommunityResource:
    """Read-only resource for communities (``graph.communities``)."""

    def __init__(self, graph_store: GraphStore) -> None:
        """Bind the resource to its owning graph's store."""
        self._store = graph_store

    async def get(self, community_id: str) -> Community | None:
        """Return one community, or ``None`` if absent."""
        row = await self._store.get_community(community_id)
        return _community_from_row(row) if row else None

    async def list(
        self, *, cursor: str | None = None, limit: int = 100
    ) -> Page[Community]:
        """Return a cursor-paginated page of communities ordered by id."""
        rows = await self._store.list_communities(cursor=cursor, limit=limit)
        items = [_community_from_row(r) for r in rows]
        return _page(items, limit)
