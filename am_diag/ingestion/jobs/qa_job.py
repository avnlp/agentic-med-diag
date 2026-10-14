"""The durable, resumable ``QAJob`` batch-evaluation engine.

Answers a batch of questions over a project, isolating failures and caching
answers so re-runs and resumes skip already-answered questions. Shares the
durable manifest/items machinery with :class:`IngestionJob` via a held
:class:`JobLedger`, and depends on the :class:`Answerer` protocol rather than on
``api.Project``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from am_diag.common.data_models.job import JobItem, JobProgress, JobRecord
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.interfaces import Answerer
from am_diag.ingestion.jobs.ledger import JobLedger


class QAJob:
    """Durable, resumable batch evaluation of questions over a project.

    Each question is answered (or served from the answer cache) and isolated:
    a failure marks its item failed without sinking the batch.

    Args:
        record: The job manifest (``kind == "qa"``).
        answerer: The project handle used to answer questions.
        cache: Content-addressed cache exposing the answer table.
        state_dir: Root state directory.
        questions: The questions to answer.
        agent_config_hash: Stable hash of the resolved agent config, mixed into
            the answer cache key so config changes invalidate cached answers.
    """

    def __init__(
        self,
        record: JobRecord,
        *,
        answerer: Answerer,
        cache: IngestionCache,
        state_dir: Path,
        questions: list[str],
        agent_config_hash: str,
    ) -> None:
        """Initialize the job and its ledger, loading any persisted item state."""
        self._ledger = JobLedger(record, state_dir=state_dir)
        self._answerer = answerer
        self._cache = cache
        self._questions = questions
        self._agent_config_hash = agent_config_hash

    @property
    def id(self) -> str:
        """Unique job id."""
        return self._ledger.job_id

    @property
    def record(self) -> JobRecord:
        """The current job manifest."""
        return self._ledger.record

    @property
    def status(self) -> str:
        """Overall job status."""
        return self._ledger.status

    @property
    def progress(self) -> JobProgress:
        """Aggregate item counters."""
        return self._ledger.progress

    def failures(self) -> list[JobItem]:
        """Return the questions that failed after exhausting retries."""
        return self._ledger.failures()

    async def run(self) -> JobRecord:
        """Answer every question, serving cache hits and isolating failures."""
        self._ledger.begin()
        sem = asyncio.Semaphore(self._ledger.record.config.answer_concurrency)
        await asyncio.gather(
            *(
                self._answer_one(question, index, sem)
                for index, question in enumerate(self._questions)
            )
        )
        self._ledger.finish()
        return self._ledger.record

    async def resume(self) -> JobRecord:
        """Resume the batch, skipping questions already answered."""
        return await self.run()

    def _answer_key(self, question: str) -> str:
        """Return the content-addressed answer cache key for a question."""
        return IngestionCache.answer_key(
            question, self._ledger.record.project_id or "", self._agent_config_hash
        )

    async def _answer_one(
        self, question: str, index: int, sem: asyncio.Semaphore
    ) -> None:
        """Answer one question (or serve the cache), isolating any failure."""
        key = self._answer_key(question)
        existing = self._ledger.get(key)
        if existing and existing.status == "succeeded":
            existing.status = "skipped"
            return
        item = self._ledger.add(
            JobItem(
                item_id=key,
                sequence_index=index,
                kind="question",
                ref={"question": question},
            )
        )
        async with sem:
            try:
                answer = await self._cache.get_answer(key)
                if answer is None:
                    result = await self._answerer.answer(
                        question, graphs=self._ledger.record.graphs
                    )
                    answer = result.model_dump()
                    await self._cache.put_answer(key, answer)
                item.status = "succeeded"
                item.output = answer
            except Exception as exc:  # noqa: BLE001
                item.status = "failed"
                item.error = {"type": type(exc).__name__, "message": str(exc)}
