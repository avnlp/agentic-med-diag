"""Ingestion pipelines — extraction, embedding, and corpus ingestion via StateGraphs."""

from __future__ import annotations

from am_diag.ingestion.corpus_ingestion import run_corpus_ingestion
from am_diag.ingestion.embedding_pipeline import build_embedding_pipeline
from am_diag.ingestion.extraction_pipeline import build_extraction_pipeline, prepare
from am_diag.ingestion.models import ExtractionReport, IngestionReport
from am_diag.ingestion.search_pipeline import build_search_pipeline


__all__ = [
    "ExtractionReport",
    "IngestionReport",
    "build_embedding_pipeline",
    "build_extraction_pipeline",
    "build_search_pipeline",
    "prepare",
    "run_corpus_ingestion",
]
