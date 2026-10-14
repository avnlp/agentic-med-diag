"""Durable, resumable ingestion and QA jobs.

Each job *holds* a :class:`JobLedger` (composition, not inheritance) and drives
its own explicit ``run`` loop. The engines depend on consumer-owned Protocols
(``IngestionTarget``, ``Answerer``) and never import ``am_diag.api``.
"""

from __future__ import annotations

from am_diag.ingestion.jobs.ingestion_job import IngestionJob
from am_diag.ingestion.jobs.ledger import JobLedger, new_job_id
from am_diag.ingestion.jobs.manager import JobManager
from am_diag.ingestion.jobs.qa_job import QAJob


__all__ = [
    "IngestionJob",
    "JobLedger",
    "JobManager",
    "QAJob",
    "new_job_id",
]
