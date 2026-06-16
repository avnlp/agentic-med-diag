"""Pipeline orchestrators — multi-corpus ingestion and multi-dataset QA runs."""

from __future__ import annotations

from am_diag.pipelines.config import IngestionConfig, QaConfig
from am_diag.pipelines.ingestion import run_all_corpus_ingestion
from am_diag.pipelines.qa import run_all_datasets


__all__ = [
    "IngestionConfig",
    "QaConfig",
    "run_all_corpus_ingestion",
    "run_all_datasets",
]
