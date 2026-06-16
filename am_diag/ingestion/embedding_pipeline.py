"""Embedding pipeline — LangGraph StateGraph.

Nodes: embed_entities → embed_relations → embed_chunks → embed_communities
→ upsert_vectors. Builds ``VectorRecord``s from ``Embeddable`` fields and
upserts into the 4 vector collections.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from am_diag.common.data_models import (
    Entity,
    Relation,
    VectorRecord,
)
from am_diag.ingestion.state import IngestionState
from am_diag.vector.embedding.base import Embedder


logger = logging.getLogger(__name__)


def build_entity_records(
    entities: list[Entity],
    chunk_text_map: dict[UUID, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Build upsert payloads, texts, and ids for entity embeddings.

    Args:
        entities: Resolved entities to embed.
        chunk_text_map: Optional mapping from chunk UUID to text.

    Returns:
        ``(payloads, texts, ids)`` tuples for the embedder.
    """
    payloads: list[dict[str, Any]] = []
    texts: list[str] = []
    ids: list[str] = []

    for ent in entities:
        text = ent.canonical_name or ent.name
        texts.append(text)
        ids.append(str(ent.id))
        payloads.append(
            {
                "id": str(ent.id),
                "name": ent.name,
                "canonical_name": ent.canonical_name or ent.name,
                "label": ent.label,
                "description": ent.description or "",
                "aliases": ent.aliases,
                "score": ent.score,
            }
        )

    return payloads, texts, ids


def build_relation_records(
    relations: list[Relation],
    chunk_text_map: dict[UUID, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Build upsert payloads, texts, and ids for relation embeddings.

    Args:
        relations: Relations to embed.
        chunk_text_map: Optional mapping from chunk UUID to text.

    Returns:
        ``(payloads, texts, ids)`` tuples for the embedder.
    """
    payloads: list[dict[str, Any]] = []
    texts: list[str] = []
    ids: list[str] = []

    for rel in relations:
        desc = rel.description or f"{rel.head_name} {rel.type} {rel.tail_name}"
        texts.append(desc)
        ids.append(str(rel.id))
        payloads.append(
            {
                "id": str(rel.id),
                "head_name": rel.head_name,
                "tail_name": rel.tail_name,
                "type": rel.type,
                "description": desc,
                "edge_text": desc,
                "score": rel.score,
            }
        )

    # Also produce type-only and full-SPO representations for richer search
    if relations:
        # Edge-type only: "TREATED_BY"
        type_payloads = [
            {"id": str(rel.id), "type": rel.type, "edge_text": rel.type}
            for rel in relations
        ]
        type_texts = [rel.type for rel in relations]
        type_ids = [str(rel.id) for rel in relations]
        payloads.extend(type_payloads)
        texts.extend(type_texts)
        ids.extend(type_ids)

        # SPO full sentence: "metformin TREATED_BY diabetes"
        spo_payloads = [
            {
                "id": str(rel.id),
                "head_name": rel.head_name,
                "tail_name": rel.tail_name,
                "type": rel.type,
                "spo_text": f"{rel.head_name} {rel.type} {rel.tail_name}",
            }
            for rel in relations
        ]
        spo_texts = [f"{rel.head_name} {rel.type} {rel.tail_name}" for rel in relations]
        spo_ids = [str(rel.id) for rel in relations]
        payloads.extend(spo_payloads)
        texts.extend(spo_texts)
        ids.extend(spo_ids)

    return payloads, texts, ids


async def _embed_node(
    state: IngestionState,
    embedder: Embedder | None,
) -> dict:
    """Embed all entity, relation, chunk, and community records."""
    if embedder is None:
        logger.warning("No embedder provided; skipping embedding.")
        return {}

    entities = state.get("entities", [])
    relations = state.get("relations", [])
    chunks = state.get("chunks", [])
    reports = state.get("reports", [])

    records: list[VectorRecord] = []

    if entities:
        payloads, texts, eids = build_entity_records(entities)
        if texts:
            vectors = await embedder.embed(texts)
            records.extend(
                VectorRecord(id=UUID(eid), vector=vec, payload=p)
                for eid, vec, p in zip(eids, vectors, payloads, strict=False)
            )

    if relations:
        payloads, texts, rids = build_relation_records(relations)
        if texts:
            vectors = await embedder.embed(texts)
            records.extend(
                VectorRecord(id=UUID(rid), vector=vec, payload=p)
                for rid, vec, p in zip(rids, vectors, payloads, strict=False)
            )

    if chunks:
        chunk_texts = [c.text for c in chunks]
        if chunk_texts:
            vectors = await embedder.embed(chunk_texts)
            records.extend(
                VectorRecord(
                    id=c.id,
                    vector=vec,
                    payload={
                        "id": str(c.id),
                        "text": c.text[:500],
                        "document_id": str(c.document_id),
                        "document_source": c.document_source,
                        "chunk_index": c.chunk_index,
                        "chunk_size": c.chunk_size,
                        "cut_type": c.cut_type,
                    },
                )
                for c, vec in zip(chunks, vectors, strict=False)
            )

    if reports:
        report_texts = [r.summary or r.title for r in reports]
        if report_texts:
            vectors = await embedder.embed(report_texts)
            records.extend(
                VectorRecord(
                    id=r.id,
                    vector=vec,
                    payload={
                        "id": str(r.id),
                        "community_id": str(r.community_id),
                        "title": r.title,
                        "summary": r.summary,
                        "report_type": r.report_type,
                    },
                )
                for r, vec in zip(reports, vectors, strict=False)
            )

    return {"records": records}


async def _upsert_node(
    state: IngestionState,
    vector_store: Any | None,
    config: dict[str, str],
) -> dict:
    """Upsert vector records to the store."""
    if vector_store is None:
        logger.warning("No vector store provided; skipping upsert.")
        return {}

    records = state.get("records", [])
    if not records:
        return {}

    entity_ids = {str(e.id) for e in state.get("entities", [])}
    relation_ids = {str(r.id) for r in state.get("relations", [])}
    chunk_ids = {str(c.id) for c in state.get("chunks", [])}

    grouped: dict[str, list[VectorRecord]] = {
        config.get("entity", "am_diag_entity"): [],
        config.get("relation", "am_diag_relation"): [],
        config.get("chunk", "am_diag_chunk"): [],
        config.get("community", "am_diag_community"): [],
    }

    for rec in records:
        sid = str(rec.id)
        if sid in entity_ids:
            grouped[config.get("entity", "am_diag_entity")].append(rec)
        elif sid in relation_ids:
            grouped[config.get("relation", "am_diag_relation")].append(rec)
        elif sid in chunk_ids:
            grouped[config.get("chunk", "am_diag_chunk")].append(rec)
        else:
            grouped[config.get("community", "am_diag_community")].append(rec)

    populated = []
    for coll, recs in grouped.items():
        if recs:
            await vector_store.upsert(coll, recs)
            populated.append(coll)

    return {"embedding_collections_populated": populated}


def build_embedding_pipeline(
    vector_store: Any | None = None,
    embedder: Embedder | None = None,
    collection_config: dict[str, str] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a compiled LangGraph StateGraph for embedding.

    Args:
        vector_store: Vector store client for upsert.
        embedder: Embedder for vectorisation.
        collection_config: Collection name mapping.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled LangGraph ``CompiledGraph``.
    """
    from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    from am_diag.ingestion.state import IngestionState  # noqa: PLC0415

    config = collection_config or {}
    builder: StateGraph = StateGraph(IngestionState)  # type: ignore

    builder.add_node("embed", lambda state: _embed_node(state, embedder))  # type: ignore[arg-type]
    builder.add_node("upsert", lambda state: _upsert_node(state, vector_store, config))  # type: ignore[arg-type]

    builder.add_edge(START, "embed")
    builder.add_edge("embed", "upsert")
    builder.add_edge("upsert", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


__all__ = [
    "build_embedding_pipeline",
    "build_entity_records",
    "build_relation_records",
]
