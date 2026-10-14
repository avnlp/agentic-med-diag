"""Tests for the durable QAJob batch-evaluation engine.

Uses a fake answerer (mocked answer) and a real temp answer cache to verify
cache-hit skipping, poison-question isolation (partial + failures), and resume
skipping of already-answered questions.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.job import JobConfig, JobRecord
from am_diag.ingestion.cache import IngestionCache
from am_diag.ingestion.jobs import QAJob


_HASH = "cfg123"


def _record() -> JobRecord:
    return JobRecord(job_id=uuid4().hex, kind="qa", project_id="p", graphs=["med"])


def _make_job(tmp_path, cache, answerer, questions, record=None):
    return QAJob(
        record or _record(),
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
        assert await cache.get_answer(
            IngestionCache.answer_key("q1", "p", _HASH, graphs=["med"])
        )

    async def test_answer_cache_hit_skips_project_answer(self, tmp_path):
        """Verify a cached answer is reused without calling the agent."""
        cache = IngestionCache(tmp_path / "med.db")
        await cache.put_answer(
            IngestionCache.answer_key("q1", "p", _HASH, graphs=["med"]),
            {"answer": "cached"},
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

    async def test_resume_from_persisted_state_in_new_job(self, tmp_path):
        """Verify a job rebuilt from disk skips questions answered earlier."""
        answerer = SimpleNamespace(
            answer=AsyncMock(return_value=ClinicalAnswer(answer="yes"))
        )
        cache = IngestionCache(tmp_path / "med.db")
        record = _record()
        await _make_job(tmp_path, cache, answerer, ["q1", "q2"], record).run()
        rebuilt = _make_job(
            tmp_path, cache, answerer, ["q1", "q2"], record.model_copy(deep=True)
        )
        result = await rebuilt.resume()
        assert result.progress.skipped == 2
        assert answerer.answer.await_count == 2

    async def test_answers_are_persisted_before_the_run_ends(self, tmp_path):
        """Verify a crash mid-run keeps the answers already completed."""

        async def answer(question, graphs=None):
            if question == "crash":
                raise asyncio.CancelledError
            return ClinicalAnswer(answer="ok")

        cache = IngestionCache(tmp_path / "med.db")
        record = JobRecord(
            job_id=uuid4().hex,
            kind="qa",
            project_id="p",
            config=JobConfig(answer_concurrency=1),
        )
        job = _make_job(
            tmp_path, cache, SimpleNamespace(answer=answer), ["q1", "crash"], record
        )
        with pytest.raises(asyncio.CancelledError):
            await job.run()
        items = (tmp_path / "jobs" / record.job_id / "items.jsonl").read_text("utf-8")
        assert '"succeeded"' in items

    async def test_answers_are_not_shared_across_graph_subsets(self, tmp_path):
        """Verify the same question over different graphs is answered again."""
        answerer = SimpleNamespace(
            answer=AsyncMock(return_value=ClinicalAnswer(answer="yes"))
        )
        cache = IngestionCache(tmp_path / "med.db")
        await _make_job(tmp_path, cache, answerer, ["q1"]).run()
        other = JobRecord(
            job_id=uuid4().hex, kind="qa", project_id="p", graphs=["drug"]
        )
        await _make_job(tmp_path, cache, answerer, ["q1"], other).run()
        assert answerer.answer.await_count == 2

    async def test_retries_then_succeeds_and_clears_error(self, tmp_path):
        """Verify a transient failure is retried within the budget."""
        calls = 0

        async def answer(question, graphs=None):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("flaky")
            return ClinicalAnswer(answer="ok")

        cache = IngestionCache(tmp_path / "med.db")
        record = _record()
        job = _make_job(tmp_path, cache, SimpleNamespace(answer=answer), ["q1"], record)
        result = await job.run()
        assert result.status == "succeeded"
        assert calls == 3
        items = (tmp_path / "jobs" / record.job_id / "items.jsonl").read_text("utf-8")
        assert '"error":null' in items

    async def test_items_expose_answers_and_errors_in_question_order(self, tmp_path):
        """Verify items() returns each question's output or error in order."""

        async def answer(question, graphs=None):
            if question == "bad":
                raise RuntimeError("boom")
            return ClinicalAnswer(answer="ok")

        cache = IngestionCache(tmp_path / "med.db")
        job = _make_job(
            tmp_path, cache, SimpleNamespace(answer=answer), ["good", "bad"]
        )
        await job.run()

        good, bad = job.items()

        assert good.ref == {"question": "good"}
        assert good.output is not None
        assert bad.error == {"type": "RuntimeError", "message": "boom"}
