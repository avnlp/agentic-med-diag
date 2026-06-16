"""Pipeline-level configuration — env-overridable with INGESTION_ / QA_ prefixes."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    """Configuration for the multi-corpus ingestion pipeline.

    All fields overridable via environment variables with the ``INGESTION_``
    prefix (e.g. ``INGESTION_BATCH_SIZE=50``).
    """

    model_config = SettingsConfigDict(
        env_prefix="INGESTION_", env_file=".env", extra="ignore"
    )

    corpus_loaders: str = "all"
    """Comma-separated corpus loader names, e.g. ``'pubmed,statpearls'``,
    or ``'all'`` to run every available loader."""

    batch_size: int = 100
    """Documents per batch passed to each corpus ingestion run."""


class QaConfig(BaseSettings):
    """Configuration for the multi-dataset QA runner.

    All fields overridable via environment variables with the ``QA_`` prefix
    (e.g. ``QA_DATASETS=careqa,medqa``).
    """

    model_config = SettingsConfigDict(env_prefix="QA_", env_file=".env", extra="ignore")

    datasets: str = "all"
    """Comma-separated dataset names, e.g. ``'careqa,medqa'``, or ``'all'``."""

    output_dir: str = "results/"
    """Directory for per-dataset JSON result files."""

    limit: int | None = None
    """Max samples per dataset (``None`` = all)."""


__all__ = [
    "IngestionConfig",
    "QaConfig",
]
