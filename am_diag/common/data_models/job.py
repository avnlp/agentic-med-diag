"""Durable job models for ingestion and batch QA.

A job is a resumable manifest of work items persisted under the state directory.
``IngestionJob`` and ``QAJob`` (the runtime engines) live in ``am_diag.ingestion``;
this module holds only the pure, serializable records they read and write.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


JobKind = Literal["ingestion", "qa"]
JobStatus = Literal["draft", "running", "succeeded", "partial", "failed", "canceled"]
ItemStatus = Literal["pending", "processing", "succeeded", "failed", "skipped"]


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class JobProgress(BaseModel):
    """Aggregate counters for a job's items.

    Attributes:
        total: Total number of items registered.
        succeeded: Items completed successfully.
        failed: Items that exhausted retries.
        skipped: Items skipped (already succeeded on resume).
        processing: Items currently in flight.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    processing: int = 0

    @property
    def percent_complete(self) -> float:
        """Fraction of items resolved (succeeded + failed + skipped), 0.0-100.0."""
        if self.total == 0:
            return 0.0
        resolved = self.succeeded + self.failed + self.skipped
        return 100.0 * resolved / self.total


class JobItem(BaseModel):
    """One unit of work (a chunk, a structured batch, or a QA question).

    Attributes:
        item_id: Stable id (content hash where applicable, else uuid).
        sequence_index: Append order.
        kind: ``"chunk"`` | ``"structured_batch"`` | ``"question"``.
        ref: Source/payload locator, for example doc id + chunk index, loader
            name + batch index, or a question-text hash.
        status: Per-item status.
        error: Structured failure detail, present on failed items.
        output: Optional per-item result payload (for example a cached answer).
    """

    item_id: str
    sequence_index: int
    kind: str
    ref: dict[str, Any] = Field(default_factory=dict)
    status: ItemStatus = "pending"
    error: dict[str, Any] | None = None
    output: dict[str, Any] | None = None


class JobConfig(BaseModel):
    """Tunable execution parameters shared by ingestion and QA jobs.

    Attributes:
        extract_concurrency: Max concurrent extraction calls.
        embed_batch_size: Number of texts embedded per batch.
        write_batch_size: Number of chunks written per UNWIND batch.
        max_retries: Per-item retry budget before marking it failed.
        fallback_model: Optional alternate model id used after retries.
        embedding_scope: Which node kinds to embed during ingestion.
        answer_concurrency: Max concurrent agent answers in a QA job.
    """

    extract_concurrency: int = 8
    embed_batch_size: int = 256
    write_batch_size: int = 5000
    max_retries: int = 3
    fallback_model: str | None = None
    embedding_scope: Literal["entities", "entities+relations"] = "entities+relations"
    answer_concurrency: int = 4


class JobRecord(BaseModel):
    """Durable manifest for an ``IngestionJob`` or ``QAJob``.

    Persisted to ``jobs/{job_id}/manifest.json``.

    Attributes:
        job_id: Unique job id.
        kind: ``"ingestion"`` or ``"qa"``.
        graph_id: Target graph (ingestion jobs).
        project_id: Target project (QA jobs).
        graphs: Federation subset for a QA job.
        config: Execution parameters.
        status: Overall job status.
        progress: Aggregate item counters.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
    """

    job_id: str
    kind: JobKind
    graph_id: str | None = None
    project_id: str | None = None
    graphs: list[str] | None = None
    config: JobConfig = Field(default_factory=JobConfig)
    status: JobStatus = "draft"
    progress: JobProgress = Field(default_factory=JobProgress)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
