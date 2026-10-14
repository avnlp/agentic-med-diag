"""The ``JobManager``: lists, loads, and resumes durable jobs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from am_diag.common.data_models.job import JobRecord


if TYPE_CHECKING:
    from am_diag.ingestion.jobs.ingestion_job import IngestionJob


# JobManager exposes a ``.list()`` method that shadows the builtin ``list``
# inside the class namespace; this alias keeps ``list[...]`` annotations valid.
_List = list


class JobManager:
    """Lists, loads, and resumes durable jobs under ``state_dir/jobs``.

    Args:
        state_dir: Root state directory; manifests live under ``state_dir/jobs``.
        live_jobs: In-session registry of live jobs, shared with the graphs that
            create them, so ``get``/``resume`` resolve a job before it persists.
    """

    def __init__(self, state_dir: Path, live_jobs: dict[str, IngestionJob]) -> None:
        """Bind the manager to a state directory and the live-job registry."""
        self._jobs_dir = state_dir / "jobs"
        self._live_jobs = live_jobs

    def list(
        self, *, kind: str | None = None, status: str | None = None
    ) -> _List[JobRecord]:
        """Return durable job manifests, optionally filtered by kind/status."""
        records: list[JobRecord] = []
        if not self._jobs_dir.exists():
            return records
        for manifest in sorted(self._jobs_dir.glob("*/manifest.json")):
            record = JobRecord.model_validate_json(manifest.read_text("utf-8"))
            if kind is not None and record.kind != kind:
                continue
            if status is not None and record.status != status:
                continue
            records.append(record)
        return records

    def get(self, job_id: str) -> JobRecord:
        """Return one job manifest.

        Raises:
            FileNotFoundError: If no manifest exists for ``job_id``.
        """
        live = self._live_jobs.get(job_id)
        if live is not None:
            return live.record
        manifest = self._jobs_dir / job_id / "manifest.json"
        return JobRecord.model_validate_json(manifest.read_text("utf-8"))

    async def resume(self, job_id: str) -> JobRecord:
        """Resume a live job created in this session.

        Raises:
            ValueError: If the job is not tracked in this session (cross-process
                resume requires re-supplying the sources via the CLI).
        """
        live = self._live_jobs.get(job_id)
        if live is None:
            raise ValueError(
                f"Job {job_id!r} is not live in this session; resume via the CLI "
                "with the original sources."
            )
        return await live.resume()
