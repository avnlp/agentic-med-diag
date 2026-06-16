"""Multi-corpus ingestion orchestrator.

Wires every ``CorpusLoader`` through the full ingestion stack
(chunking → extraction → Neo4j → vector embedding) in sequence.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from am_diag.db.graph.neo4j import Neo4jClient, create_neo4j_client
from am_diag.db.vector.base import VectorStoreBase
from am_diag.db.vector.qdrant import create_qdrant_store
from am_diag.ingestion.corpus_ingestion import run_corpus_ingestion
from am_diag.ingestion.models import IngestionReport
from am_diag.loaders.corpus import (
    ClinicalGuidelinesCorpusLoader,
    CorpusLoader,
    PubmedCaseReportsCorpusLoader,
    PubMedCorpusLoader,
    StatPearlsCorpusLoader,
    TextbooksCorpusLoader,
)
from am_diag.pipelines.config import IngestionConfig
from am_diag.vector.embedding import Embedder
from am_diag.vector.embedding.sentence_transformers import (
    SentenceTransformersEmbedder,
)


logger = logging.getLogger(__name__)


CORPUS_LOADER_REGISTRY: dict[str, type[CorpusLoader]] = {
    "clinical_guidelines": ClinicalGuidelinesCorpusLoader,
    "pubmed": PubMedCorpusLoader,
    "pubmed_case_reports": PubmedCaseReportsCorpusLoader,
    "statpearls": StatPearlsCorpusLoader,
    "textbooks": TextbooksCorpusLoader,
}


@dataclass
class _LimitedLoader:
    """Wraps a ``CorpusLoader`` to cap total documents via ``astream(limit=…)``."""

    _loader: CorpusLoader
    _limit: int | None

    async def astream(
        self,
        batch_size: int = 100,
        offset: int = 0,
        limit: int | None = None,
    ) -> AsyncIterator[list[Any]]:
        effective = self._limit or limit
        async for batch in self._loader.astream(
            batch_size=batch_size,
            offset=offset,
            limit=effective,
        ):
            yield batch


def _resolve_loaders(
    spec: str,
) -> list[type[CorpusLoader]]:
    if spec == "all":
        return list(CORPUS_LOADER_REGISTRY.values())
    names = [n.strip() for n in spec.split(",")]
    return [CORPUS_LOADER_REGISTRY[n] for n in names]


async def run_all_corpus_ingestion(
    graph_store: Neo4jClient,
    vector_store: VectorStoreBase,
    embedder: Embedder,
    config: IngestionConfig | None = None,
) -> list[IngestionReport]:
    """Run the full ingestion stack for every corpus loader in ``config``.

    Args:
        graph_store: Graph database client (Neo4j).
        vector_store: Vector store client (Qdrant / Weaviate).
        embedder: Embedder for vectorisation.
        config: Ingestion configuration.  Defaults to ``IngestionConfig()``.

    Returns:
        One ``IngestionReport`` per corpus loader.
    """
    cfg = config or IngestionConfig()
    loader_classes = _resolve_loaders(cfg.corpus_loaders)

    reports: list[IngestionReport] = []
    for cls in loader_classes:
        name = getattr(cls, "corpus_name", cls.__name__)
        logger.info("Starting ingestion for corpus: %s", name)

        loader: CorpusLoader = cls()
        if cfg.batch_size:
            pass  # passed directly to run_corpus_ingestion below

        report = await run_corpus_ingestion(
            corpus_loader=loader,
            graph_store=graph_store,
            vector_store=vector_store,
            embedder=embedder,
            batch_size=cfg.batch_size,
        )
        reports.append(report)

        logger.info(
            "Finished %s: %d docs, %d chunks, %d entities, %d relations",
            name,
            report.documents_ingested,
            report.chunks_ingested,
            report.entities_ingested,
            report.relations_ingested,
        )

    return reports


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest one or more medical corpora into Neo4j + vector store.",
    )
    parser.add_argument(
        "--corpus",
        default="all",
        help=(
            "Comma-separated corpus names, or 'all' (default). "
            f"Options: {', '.join(CORPUS_LOADER_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Documents per batch (default: 100).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (``am-diag-ingest``).

    Usage::

        uv run am-diag-ingest
        uv run am-diag-ingest --corpus pubmed,statpearls --batch-size 50
    """
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = IngestionConfig(
        corpus_loaders=args.corpus,
        batch_size=args.batch_size,
    )

    async def _run() -> list[IngestionReport]:
        async with (
            create_neo4j_client() as graph_store,
            create_qdrant_store() as vector_store,
        ):
            embedder: Embedder = SentenceTransformersEmbedder()
            return await run_all_corpus_ingestion(
                graph_store=graph_store,
                vector_store=vector_store,
                embedder=embedder,
                config=config,
            )

    reports = asyncio.run(_run())

    print("\n=== Ingestion Summary ===")
    for r in reports:
        print(
            f"  {r.batch_id or 'corpus'}: "
            f"{r.documents_ingested} docs, "
            f"{r.chunks_ingested} chunks, "
            f"{r.entities_ingested} entities, "
            f"{r.relations_ingested} relations"
        )


__all__ = [
    "CORPUS_LOADER_REGISTRY",
    "IngestionReport",
    "main",
    "run_all_corpus_ingestion",
]
