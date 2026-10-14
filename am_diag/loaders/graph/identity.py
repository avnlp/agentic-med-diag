"""Deterministic id helpers for structured-graph loaders.

Structured biomedical graphs (OptimusKG and per-source subgraphs) key on CURIEs.
These helpers derive stable UUIDs from CURIEs so the same node/edge produced by
different loaders or across streaming batches merges into one Neo4j node/edge
(``MERGE`` on the deterministic id), without disturbing the name-keyed identity
of text-extracted entities.
"""

from __future__ import annotations

from uuid import NAMESPACE_OID, UUID, uuid5


def curie_to_entity_id(curie: str) -> UUID:
    """Return a stable ``Entity`` id derived from a CURIE.

    The same CURIE always maps to the same UUID, so entities sharing a CURIE
    across loaders or batches resolve to a single node under ``MERGE``.

    Args:
        curie: Canonical external identifier, e.g. ``"MONDO:0005148"``.

    Returns:
        Deterministic ``UUID5`` for the entity.
    """
    return uuid5(NAMESPACE_OID, f"Entity:curie:{curie.strip()}")


def relation_id(head_curie: str, rel_type: str, tail_curie: str) -> UUID:
    """Return a stable ``Relation`` id for one ``(head, type, tail)`` edge.

    Makes re-ingestion idempotent: a repeated edge merges onto the same
    ``RELATES_TO`` relationship instead of creating a duplicate.

    Args:
        head_curie: Head entity CURIE.
        rel_type: Semantic relation type, e.g. ``"EXPRESSED_IN"``.
        tail_curie: Tail entity CURIE.

    Returns:
        Deterministic ``UUID5`` for the edge.
    """
    return uuid5(
        NAMESPACE_OID,
        f"Relation:{head_curie.strip()}|{rel_type}|{tail_curie.strip()}",
    )
