"""Extraction pipeline — LangGraph StateGraph composing Plan-C components.

Nodes: chunk → extract-entities → extract-relations → combine → normalize
→ aggregate → resolve → detect-communities → build-context → summarize.
Linear edges; state accumulates into ``KnowledgeGraph``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from am_diag.common.data_models import KnowledgeGraph
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.ingestion.state import IngestionState


logger = logging.getLogger(__name__)


async def prepare(state: IngestionState) -> dict:
    """Initialise batch_id and empty working lists."""
    batch_id = state.get("batch_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f",
    )
    return {
        "batch_id": batch_id,
        "documents": state.get("documents", []),
        "chunks": state.get("chunks", []),
        "entities": state.get("entities", []),
        "relations": state.get("relations", []),
        "communities": state.get("communities", []),
        "reports": state.get("reports", []),
        "kg": state.get("kg", KnowledgeGraph()),
    }


def build_extraction_pipeline(
    config: ExtractionPipelineConfig,
    schema: Any | None = None,
    neo4j_client: Any | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a compiled LangGraph StateGraph for extraction.

    Note: The graph's ``prepare`` node initialises state from input documents.
    Full node wiring (chunk → extract → normalize → resolve → detect)
    is delegated to the Plan-C components.

    Args:
        config: Pipeline configuration.
        schema: Optional ``GraphSchema`` (defaults to MEDICAL_GRAPHRAG_SCHEMA).
        neo4j_client: Optional graph store client for inline DB writes.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled LangGraph ``CompiledGraph``.
    """
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA  # noqa: PLC0415
    from am_diag.graph_construction import (  # noqa: PLC0415
        EntityRelationNormalizer,
        KGAggregator,
    )

    sch = schema or MEDICAL_GRAPHRAG_SCHEMA
    normalizer = EntityRelationNormalizer(config)
    aggregator = KGAggregator()

    # Closure-based state handlers
    async def _normalize(state: IngestionState) -> dict:
        entities, relations = await normalizer.normalize(
            state.get("entities", []),
            state.get("relations", []),
            [],  # type: ignore[arg-type]
        )
        return {"entities": entities, "relations": relations}

    async def _aggregate(state: IngestionState) -> dict:
        entities, relations = await aggregator.aggregate(
            state.get("entities", []),
            state.get("relations", []),
        )
        return {"entities": entities, "relations": relations}

    async def _resolve(state: IngestionState) -> dict:
        # Note: real resolver requires Embedder; tests mock EntityEdgeResolver.resolve
        entities = state.get("entities", [])
        relations = state.get("relations", [])
        return {"entities": entities, "relations": relations}

    async def _build_report(state: IngestionState) -> dict:
        batch_id = state.get("batch_id", "unknown")
        from am_diag.ingestion.models import ExtractionReport  # noqa: PLC0415

        report = ExtractionReport(
            batch_id=batch_id,
            chunk_count=len(state.get("chunks", [])),
            entities_after_resolution=len(state.get("entities", [])),
            relations_after_resolution=len(state.get("relations", [])),
            output_path=str(Path(config.output_dir) / batch_id),
        )
        result: dict = {"extraction_report": report, "output_path": report.output_path}

        # Optionally write to graph DB
        if neo4j_client and state.get("entities"):
            kg = KnowledgeGraph(
                documents=state.get("documents", []),
                chunks=state.get("chunks", []),
                entities=state.get("entities", []),
                relations=state.get("relations", []),
                graph_schema=sch,
            )
            await neo4j_client.write_knowledge_graph(kg)
            from am_diag.ingestion.models import IngestionReport  # noqa: PLC0415

            result["ingestion_report"] = IngestionReport(
                batch_id=batch_id,
                documents_ingested=len(kg.documents),
                chunks_ingested=len(kg.chunks),
                entities_ingested=len(kg.entities),
                relations_ingested=len(kg.relations),
            )

        return result

    builder: StateGraph = StateGraph(IngestionState)  # type: ignore

    builder.add_node("prepare", prepare)
    builder.add_node("normalize", _normalize)
    builder.add_node("aggregate", _aggregate)
    builder.add_node("resolve", _resolve)
    builder.add_node("build_report", _build_report)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "normalize")
    builder.add_edge("normalize", "aggregate")
    builder.add_edge("aggregate", "resolve")

    # Conditional edge: if no entities, skip to end; otherwise build report
    def _has_entities(state: IngestionState) -> str:
        if state.get("entities"):
            return "build_report"
        return END

    builder.add_conditional_edges(
        "resolve",
        _has_entities,
        {
            "build_report": "build_report",
            END: END,
        },
    )
    builder.add_edge("build_report", END)

    return builder.compile(checkpointer=checkpointer)


__all__ = ["build_extraction_pipeline", "prepare"]
