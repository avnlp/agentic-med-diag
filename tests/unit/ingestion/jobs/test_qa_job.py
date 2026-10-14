"""Tests for the durable QAJob batch-evaluation engine.

Uses a fake answerer (mocked answer) and a real temp answer cache to verify
cache-hit skipping, poison-question isolation (partial + failures), and resume
skipping of already-answered questions.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.job import JobRecord
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.jobs import QAJob


_HASH = "cfg123"


def _record() -> JobRecord:
    return JobRecord(job_id=uuid4().hex, kind="qa", project_id="p", graphs=["med"])


def _make_job(tmp_path, cache, answerer, questions):
    return QAJob(
        _record(),
        answerer=answerer,
        cache=cache,
        state_dir=tmp_path,
        questions=questions,
        agent_config_hash=_HASH,
    )


class TestQAJob:
    """Tests for QAJob.run and resume."""

    async def test_run_answers_and_caches(self, tmp_path):
        """Verify a clean run answers each question and caches the result."""
        answerer = SimpleNamespace(
            answer=AsyncMock(return_value=ClinicalAnswer(answer="yes"))
        )
        cache = IngestionCache(tmp_path / "med.db")
        job = _make_job(tmp_path, cache, answerer, ["q1", "q2"])
        record = await job.run()
        assert record.status == "succeeded"
        assert record.progress.succeeded == 2
        assert answerer.answer.await_count == 2
        assert await cache.get_answer(IngestionCache.answer_key("q1", "p", _HASH))

    async def test_answer_cache_hit_skips_project_answer(self, tmp_path):
        """Verify a cached answer is reused without calling the agent."""
        cache = IngestionCache(tmp_path / "med.db")
        await cache.put_answer(
            IngestionCache.answer_key("q1", "p", _HASH), {"answer": "cached"}
        )
        answerer = SimpleNamespace(answer=AsyncMock())
        job = _make_job(tmp_path, cache, answerer, ["q1"])
        record = await job.run()
        assert record.status == "succeeded"
        answerer.answer.assert_not_called()

    async def test_poison_question_yields_partial(self, tmp_path):
        """Verify a failing question is isolated and the job ends partial."""

        async def answer(question, graphs=None):
            if question == "bad":
                raise RuntimeError("boom")
            return ClinicalAnswer(answer="ok")

        answerer = SimpleNamespace(answer=answer)
        cache = IngestionCache(tmp_path / "med.db")
        job = _make_job(tmp_path, cache, answerer, ["good", "bad"])
        record = await job.run()
        assert record.status == "partial"
        assert record.progress.failed == 1
        assert len(job.failures()) == 1

    async def test_resume_skips_answered(self, tmp_path):
        """Verify resuming skips already-answered questions."""
        answerer = SimpleNamespace(
            answer=AsyncMock(return_value=ClinicalAnswer(answer="yes"))
        )
        cache = IngestionCache(tmp_path / "med.db")
        job = _make_job(tmp_path, cache, answerer, ["q1", "q2"])
        await job.run()
        assert answerer.answer.await_count == 2
        record = await job.resume()
        assert record.progress.skipped == 2
        assert answerer.answer.await_count == 2
