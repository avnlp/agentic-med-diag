"""End-to-end corpus ingestion.

Orchestrates: corpus loader → chunker → extraction pipeline → embedding
pipeline → graph store write. Returns an ``IngestionReport``.
"""

from __future__ import annotations

import logging
from typing import Any

from am_diag.common.data_models import KnowledgeGraph, VectorRecord
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.ingestion.models import IngestionReport
from am_diag.retrieval.config import RetrievalConfig
from am_diag.vector.embedding.base import Embedder


logger = logging.getLogger(__name__)


async def _embed_and_upsert(
    embedder: Embedder,
    vector_store: VectorStoreBase,
    rconfig: RetrievalConfig,
    extracted_entities: list[Any],
    extracted_relations: list[Any],
    all_chunks: list[Any],
) -> None:
    """Embed entities, relations, and chunks; upsert vectors to store."""
    if not extracted_entities and not extracted_relations:
        return

    entity_texts: list[str] = []
    entity_payloads: list[dict[str, Any]] = []
    relation_texts: list[str] = []
    relation_payloads: list[dict[str, Any]] = []

    for ent in extracted_entities:
        text = ent.canonical_name or ent.name
        entity_texts.append(text)
        entity_payloads.append(
            {
                "id": str(ent.id),
                "name": ent.name,
                "canonical_name": ent.canonical_name or ent.name,
                "label": ent.label,
            }
        )

    for rel in extracted_relations:
        text = rel.description or f"{rel.head_name} {rel.type} {rel.tail_name}"
        relation_texts.append(text)
        relation_payloads.append(
            {
                "id": str(rel.id),
                "head_name": rel.head_name,
                "tail_name": rel.tail_name,
                "type": rel.type,
            }
        )

    if entity_texts:
        entity_vecs = await embedder.embed(entity_texts)
        entity_records = [
            VectorRecord(id=ent.id, vector=vec, payload=pl)
            for ent, vec, pl in zip(
                extracted_entities,
                entity_vecs,
                entity_payloads,
                strict=False,
            )
        ]
        await vector_store.upsert(rconfig.entity_collection, entity_records)

    if relation_texts:
        relation_vecs = await embedder.embed(relation_texts)
        relation_records = [
            VectorRecord(id=rel.id, vector=vec, payload=pl)
            for rel, vec, pl in zip(
                extracted_relations,
                relation_vecs,
                relation_payloads,
                strict=False,
            )
        ]
        await vector_store.upsert(rconfig.relation_collection, relation_records)

    chunk_texts = [c.text for c in all_chunks]
    if chunk_texts:
        chunk_vecs = await embedder.embed(chunk_texts)
        chunk_records = [
            VectorRecord(
                id=c.id,
                vector=vec,
                payload={
                    "id": str(c.id),
                    "text": c.text[:500],
                    "document_id": str(c.document_id),
                    "document_source": c.document_source,
                },
            )
            for c, vec in zip(all_chunks, chunk_vecs, strict=False)
        ]
        await vector_store.upsert(rconfig.chunk_collection, chunk_records)


async def run_corpus_ingestion(
    corpus_loader: Any,
    graph_store: GraphStore,
    vector_store: VectorStoreBase,
    embedder: Embedder,
    extraction_config: ExtractionPipelineConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    batch_size: int = 100,
) -> IngestionReport:
    """Stream documents from ``corpus_loader`` through the full ingestion stack.

    1. Stream ``Document`` batches from the corpus loader.
    2. Chunk each document via the configured chunker.
    3. Run extraction pipeline (entities → relations → resolution).
    4. Run embedding pipeline (embed → upsert vectors).
    5. Write to graph store via ``write_knowledge_graph``.

    Args:
        corpus_loader: ``CorpusLoader`` instance (async streaming).
        graph_store: Graph database client.
        vector_store: Vector store client.
        embedder: Embedder for vectorisation.
        extraction_config: Extraction pipeline config.
        retrieval_config: Retrieval config (collection names, top-k, etc.).
        batch_size: Max documents per batch.

    Returns:
        ``IngestionReport`` summarising what was ingested.
    """
    from am_diag.chunking.recursive_character_text_splitter import (  # noqa: PLC0415
        RecursiveCharacterTextChunker,
    )
    from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA  # noqa: PLC0415
    from am_diag.ingestion.extraction_pipeline import (  # noqa: PLC0415
        build_extraction_pipeline,
    )

    econfig = extraction_config or ExtractionPipelineConfig()
    rconfig = retrieval_config or RetrievalConfig()

    # Build pipelines
    extraction_app = build_extraction_pipeline(
        econfig,
        schema=MEDICAL_GRAPHRAG_SCHEMA,
        neo4j_client=graph_store,
    )

    chunker = RecursiveCharacterTextChunker()

    total_report = IngestionReport(batch_id="corpus_ingestion")
    batch_id = 0

    async for doc_batch in corpus_loader.astream(batch_size=batch_size):
        batch_id += 1
        bid = f"batch_{batch_id:04d}"

        # Chunk
        all_chunks: list[Any] = []
        for doc in doc_batch:
            chunks = await chunker.chunk([doc])
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.info("Batch %s: no chunks produced, skipping.", bid)
            continue

        # Run extraction pipeline
        result = await extraction_app.ainvoke(
            {
                "batch_id": bid,
                "documents": doc_batch,
                "chunks": all_chunks,
            }
        )

        extracted_entities = result.get("entities", [])
        extracted_relations = result.get("relations", [])

        # Build KnowledgeGraph for graph store write
        kg = KnowledgeGraph(
            documents=doc_batch,
            chunks=all_chunks,
            entities=extracted_entities,
            relations=extracted_relations,
            graph_schema=MEDICAL_GRAPHRAG_SCHEMA,
        )

        # Write to graph store
        await graph_store.write_knowledge_graph(kg)

        # Embed and upsert vectors
        await _embed_and_upsert(
            embedder=embedder,
            vector_store=vector_store,
            rconfig=rconfig,
            extracted_entities=extracted_entities,
            extracted_relations=extracted_relations,
            all_chunks=all_chunks,
        )

        # Accumulate totals
        total_report.documents_ingested += len(doc_batch)
        total_report.chunks_ingested += len(all_chunks)
        total_report.entities_ingested += len(extracted_entities)
        total_report.relations_ingested += len(extracted_relations)

        logger.info(
            "Batch %s: %d docs, %d chunks, %d entities, %d relations",
            bid,
            len(doc_batch),
            len(all_chunks),
            len(extracted_entities),
            len(extracted_relations),
        )

    return total_report


__all__ = ["run_corpus_ingestion"]
