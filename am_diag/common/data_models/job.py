"""Durable job models for ingestion and batch QA.

A job is a resumable manifest of work items persisted under the state directory.
``IngestionJob`` and ``QAJob`` (the runtime engines) live in ``am_diag.ingestion``;
this module holds only the pure, serializable records they read and write.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


JobKind = Literal["ingestion", "qa"]
JobStatus = Literal["draft", "running", "succeeded", "partial", "failed", "canceled"]
ItemStatus = Literal["pending", "processing", "succeeded", "failed", "skipped"]

_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def validate_job_id(job_id: str) -> str:
    """Return ``job_id`` if it is a single safe path component.

    Raises:
        ValueError: If the id is empty or contains anything but letters, digits,
            ``_`` and ``-`` (so it cannot escape the jobs directory).
    """
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError(f"Invalid job id {job_id!r}: use letters, digits, _ and -")
    return job_id


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
        max_retries: Per-item attempt budget before marking it failed (at least 1).
        embedding_scope: Which node kinds to embed during ingestion.
        answer_concurrency: Max concurrent agent answers in a QA job.
    """

    extract_concurrency: int = Field(default=8, ge=1)
    embed_batch_size: int = Field(default=256, ge=1)
    write_batch_size: int = Field(default=5000, ge=1)
    max_retries: int = Field(default=3, ge=1)
    embedding_scope: Literal["entities", "entities+relations"] = "entities+relations"
    answer_concurrency: int = Field(default=4, ge=1)


class JobRecord(BaseModel):
    """Durable manifest for an ``IngestionJob`` or ``QAJob``.

    Persisted to ``jobs/{job_id}/manifest.json``.

    Attributes:
        job_id: Unique job id.
        kind: ``"ingestion"`` or ``"qa"``.
        graph_id: Target graph (ingestion jobs).
        project_id: Target project (QA jobs).
        graphs: Federation subset for a QA job.
        finalize: Whether an ingestion job must finalize communities.
        finalized: Whether finalization has completed.
        config: Execution parameters.
        status: Overall job status.
        progress: Aggregate item counters.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.

    Raises:
        ValueError: If ``job_id`` is not a safe path component, an ingestion job
            has no ``graph_id``, or a QA job has no ``project_id``.
    """

    job_id: str
    kind: JobKind
    graph_id: str | None = None
    project_id: str | None = None
    graphs: list[str] | None = None
    finalize: bool = False
    finalized: bool = False
    config: JobConfig = Field(default_factory=JobConfig)
    status: JobStatus = "draft"
    progress: JobProgress = Field(default_factory=JobProgress)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("job_id")
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        return validate_job_id(value)

    @model_validator(mode="after")
    def _check_target(self) -> JobRecord:
        if self.kind == "ingestion" and not self.graph_id:
            raise ValueError("Ingestion jobs require a graph_id")
        if self.kind == "qa" and not self.project_id:
            raise ValueError("QA jobs require a project_id")
        return self
