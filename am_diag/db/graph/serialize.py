"""Generic DataPoint → Neo4j parameter serializer.

Converts DataPoint subclasses to the flat dict shapes expected by the
.cypher templates in common/cypher/. UUID fields become strings; nested
models and dicts become JSON strings; all other values pass through.

Each public function corresponds to one .cypher template's expected
``{id, props}`` or ``{id, head_id, tail_id, props}`` record shape.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from am_diag.common.data_models.base import DataPoint
from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport
from am_diag.common.data_models.document import Document
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation


def _json_dumps(v: Any) -> str:
    """Serialize a value to JSON, handling UUID objects.

    Pydantic ``model_dump()`` in Python mode preserves UUID objects as
    ``uuid.UUID`` instances, which ``json.dumps`` cannot serialize natively.
    ``default=str`` coerces UUIDs (and any other non-serializable types) to
    their string representation.

    Args:
        v: The value to serialize (typically a dict or list).

    Returns:
        A JSON string with UUIDs converted to strings.
    """
    return json.dumps(v, default=str)


def _serialize_value(v: Any) -> Any:
    """Convert a single value to a Neo4j-safe scalar.

    Neo4j cannot store Python UUIDs, Pydantic models, or arbitrary dicts
    natively. This function applies the minimum transformation needed:
    UUIDs become strings, nested models/dicts/lists become JSON strings,
    and all other primitives are returned unchanged.

    Args:
        v: The value to convert.

    Returns:
        A Neo4j-compatible scalar, list, or JSON string.
    """
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, BaseModel):
        return v.model_dump_json()
    if isinstance(v, (dict, list)):
        return _json_dumps(v)
    return v


def to_params(dp: DataPoint, *, drop: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Serialize a DataPoint to a flat dict for Neo4j Cypher parameters.

    UUID fields become strings; nested BaseModels and dicts/lists become
    JSON strings; primitives pass through. Fields in ``drop`` are excluded.

    Args:
        dp: The DataPoint to serialize.
        drop: Field names to exclude from the output dict.

    Returns:
        A flat dict safe to use as Neo4j Cypher parameters.
    """
    raw = dp.model_dump(exclude=set(drop))
    return {k: _serialize_value(v) for k, v in raw.items()}


def document_record(doc: Document) -> dict[str, Any]:
    """Build the ``{id, props}`` record shape expected by upsert_documents.cypher.

    Args:
        doc: The Document DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(doc)
    doc_id = params.pop("id")
    return {"id": doc_id, "props": params}


def chunk_record(chunk: Chunk) -> dict[str, Any]:
    """Build the ``{id, props}`` record shape expected by upsert_chunks.cypher.

    Args:
        chunk: The Chunk DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(chunk)
    chunk_id = params.pop("id")
    return {"id": chunk_id, "props": params}


def entity_record(entity: Entity) -> dict[str, Any]:
    """Build the ``{id, props, type_label}`` record shape for upsert_entities.cypher.

    ``type_label`` is extracted separately so the client can group entities by
    label and render the cypher template once per type group — the template
    receives it as a backtick-escaped identifier, not a Cypher parameter.

    Args:
        entity: The Entity DataPoint to serialize.

    Returns:
        Dict with ``id``, ``type_label`` (str), and ``props`` (flat param dict).
    """
    params = to_params(entity)
    entity_id = params.pop("id")
    type_label = params.pop("label")  # rendered into Cypher, not a $parameter
    return {"id": entity_id, "props": params, "type_label": type_label}


def relation_record(rel: Relation) -> dict[str, Any]:
    """Build the ``{id, head_id, tail_id, props}`` record for upsert_relations.cypher.

    ``head_id`` and ``tail_id`` are pulled out at the top level so the Cypher
    template can MATCH the endpoint nodes by id in a single UNWIND pass.

    Args:
        rel: The Relation DataPoint to serialize.

    Returns:
        Dict with ``id``, ``head_id``, ``tail_id`` (all str), and ``props``.
    """
    params = to_params(rel)
    rel_id = params.pop("id")
    head_id = params.pop("head_id")
    tail_id = params.pop("tail_id")
    return {
        "id": rel_id,
        "head_id": head_id,
        "tail_id": tail_id,
        "props": params,
    }


def community_record(community: Community) -> dict[str, Any]:
    """Build the ``{id, props}`` record for upsert_communities.cypher.

    Args:
        community: The Community DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(community)
    c_id = params.pop("id")
    return {"id": c_id, "props": params}


def report_record(report: CommunityReport) -> dict[str, Any]:
    """Build the ``{id, props}`` record for upsert_reports.cypher.

    Args:
        report: The CommunityReport DataPoint to serialize.

    Returns:
        Dict with ``id`` (str) and ``props`` (flat param dict).
    """
    params = to_params(report)
    r_id = params.pop("id")
    return {"id": r_id, "props": params}


def has_entity_records(entities: list[Entity]) -> list[dict[str, str]]:
    """Build the ``[{entity_id, chunk_id}]`` records for link_has_entity.cypher.

    One record is produced per (entity, chunk_id) pair because a single entity
    may appear in multiple chunks and the cypher template merges each edge
    independently.

    Args:
        entities: Entities whose provenance chunk_ids should be linked.

    Returns:
        List of dicts each containing ``entity_id`` and ``chunk_id`` as strings.
    """
    records = []
    for entity in entities:
        for chunk_id in entity.provenance.chunk_ids:
            records.append(
                {
                    "entity_id": str(entity.id),
                    "chunk_id": str(chunk_id),
                }
            )
    return records


def in_community_records(communities: list[Community]) -> list[dict[str, str]]:
    """Build the ``[{entity_id, community_id}]`` records for link_in_community.cypher.

    One record per (entity, community) pair; the cypher template merges the
    IN_COMMUNITY edge for each pair independently.

    Args:
        communities: Communities whose entity membership should be linked.

    Returns:
        List of dicts with ``entity_id`` and ``community_id`` as strings.
    """
    records = []
    for community in communities:
        for entity_id in community.entity_ids:
            records.append(
                {
                    "entity_id": str(entity_id),
                    "community_id": str(community.id),
                }
            )
    return records


def parent_community_records(communities: list[Community]) -> list[dict[str, str]]:
    """Build the ``[{child_id, parent_id}]`` records for link_parent_community.cypher.

    Only communities with a non-None ``parent`` UUID are included; top-level
    communities have no parent edge and are silently skipped.

    Args:
        communities: Communities to inspect for parent links.

    Returns:
        List of dicts with ``child_id`` and ``parent_id`` as strings.
    """
    return [
        {"child_id": str(c.id), "parent_id": str(c.parent)}
        for c in communities
        if c.parent is not None
    ]


def has_report_records(reports: list[CommunityReport]) -> list[dict[str, str]]:
    """Build the ``[{community_id, report_id}]`` records for link_has_report.cypher.

    Args:
        reports: Community reports whose community links should be materialized.

    Returns:
        List of dicts with ``community_id`` and ``report_id`` as strings.
    """
    return [
        {"community_id": str(r.community_id), "report_id": str(r.id)} for r in reports
    ]


__all__ = [
    "to_params",
    "document_record",
    "chunk_record",
    "entity_record",
    "relation_record",
    "community_record",
    "report_record",
    "has_entity_records",
    "in_community_records",
    "parent_community_records",
    "has_report_records",
]
