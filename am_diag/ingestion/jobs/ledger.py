"""The ``JobLedger``: a job's durable record of items and status.

A job *holds* a ledger and drives its own ``run`` loop; there is no job base
class. The ledger owns the :class:`JobRecord` plus per-item state under
``state_dir/jobs/{job_id}/`` (``manifest.json`` + ``items.jsonl``), recomputes
progress, and rewrites both files as a crash-safe resume point. This mirrors how
mature ingestion libraries compose durable state rather than subclassing.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from am_diag.common.data_models.job import JobItem, JobProgress, JobRecord


def new_job_id() -> str:
    """Return a fresh unique job id."""
    return uuid4().hex


def _compute_progress(items: dict[str, JobItem]) -> JobProgress:
    """Recompute aggregate progress counters from item statuses."""
    progress = JobProgress(total=len(items))
    for item in items.values():
        if item.status == "succeeded":
            progress.succeeded += 1
        elif item.status == "failed":
            progress.failed += 1
        elif item.status == "skipped":
            progress.skipped += 1
        elif item.status == "processing":
            progress.processing += 1
    return progress


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so readers never see a partial file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class JobLedger:
    """Durable record of a job's items and status.

    Loads any persisted item state on construction so a resumed job sees prior
    progress. ``persist`` recounts progress and atomically rewrites the manifest
    and items file; ``begin`` / ``finish`` encapsulate the start and terminal
    status transitions every job shares.
    """

    def __init__(self, record: JobRecord, *, state_dir: Path) -> None:
        """Bind the ledger to a record and load persisted state, if any."""
        self._record = record
        self._dir = state_dir / "jobs" / record.job_id
        self._restore_finalization()
        self._items: dict[str, JobItem] = self._load()

    def _restore_finalization(self) -> None:
        """Carry the persisted finalization state onto a re-created record."""
        manifest = self._dir / "manifest.json"
        if not manifest.exists():
            return
        persisted = JobRecord.model_validate_json(manifest.read_text("utf-8"))
        self._record.finalize = self._record.finalize or persisted.finalize
        self._record.finalized = self._record.finalized or persisted.finalized

    def _load(self) -> dict[str, JobItem]:
        """Load persisted item state from ``items.jsonl``, if present."""
        items: dict[str, JobItem] = {}
        items_file = self._dir / "items.jsonl"
        if not items_file.exists():
            return items
        for line in items_file.read_text("utf-8").splitlines():
            if line.strip():
                item = JobItem.model_validate_json(line)
                items[item.item_id] = item
        return items

    @property
    def record(self) -> JobRecord:
        """The current job manifest."""
        return self._record

    @property
    def job_id(self) -> str:
        """Unique job id."""
        return self._record.job_id

    @property
    def status(self) -> str:
        """Overall job status."""
        return self._record.status

    @property
    def progress(self) -> JobProgress:
        """Aggregate item counters (recomputed on every ``persist``)."""
        return self._record.progress

    @property
    def items(self) -> dict[str, JobItem]:
        """The per-item state map keyed by item id."""
        return self._items

    def get(self, item_id: str) -> JobItem | None:
        """Return the tracked item for ``item_id``, or ``None``."""
        return self._items.get(item_id)

    def add(self, item: JobItem) -> JobItem:
        """Track ``item`` if absent and return the stored item (idempotent)."""
        return self._items.setdefault(item.item_id, item)

    def failures(self) -> list[JobItem]:
        """Return the items that failed after exhausting retries."""
        return [i for i in self._items.values() if i.status == "failed"]

    def begin(self) -> None:
        """Mark the job running and persist (the first resume point)."""
        self._record.status = "running"
        self.persist()

    def finish(self) -> None:
        """Set the terminal status from failures and persist."""
        self._record.progress = _compute_progress(self._items)
        self._record.status = "partial" if self._record.progress.failed else "succeeded"
        self.persist()

    def persist(self) -> None:
        """Recount progress and atomically rewrite the manifest + items file."""
        self._record.progress = _compute_progress(self._items)
        self._record.updated_at = datetime.now(timezone.utc)
        self._dir.mkdir(parents=True, exist_ok=True)
        lines = [
            item.model_dump_json()
            for item in sorted(self._items.values(), key=lambda i: i.sequence_index)
        ]
        _write_atomic(self._dir / "items.jsonl", "\n".join(lines))
        _write_atomic(
            self._dir / "manifest.json", self._record.model_dump_json(indent=2)
        )
