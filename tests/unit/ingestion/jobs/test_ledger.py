"""Tests for the JobLedger durable-state component.

Verifies the manifest/items persistence, progress recount, terminal status
selection, and that a fresh ledger over the same directory restores items (the
resume substrate both job types share).
"""

from __future__ import annotations

import json

from am_diag.common.data_models.job import JobItem, JobRecord
from am_diag.ingestion.jobs.ledger import JobLedger, new_job_id


def _record() -> JobRecord:
    return JobRecord(job_id=new_job_id(), kind="ingestion", graph_id="med")


def _record_with_id(job_id: str) -> JobRecord:
    return JobRecord(job_id=job_id, kind="ingestion", graph_id="med")


def _item(item_id: str, *, status: str = "pending", seq: int = 0) -> JobItem:
    return JobItem(item_id=item_id, sequence_index=seq, kind="chunk", status=status)


class TestJobLedger:
    """Tests for JobLedger persistence, recount, and resume loading."""

    def test_begin_persists_running_manifest(self, tmp_path):
        """begin() marks running and writes the manifest resume point."""
        ledger = JobLedger(_record(), state_dir=tmp_path)
        ledger.begin()
        assert ledger.status == "running"
        manifest = tmp_path / "jobs" / ledger.job_id / "manifest.json"
        assert json.loads(manifest.read_text("utf-8"))["status"] == "running"

    def test_add_is_idempotent(self, tmp_path):
        """add() returns the first stored item for a repeated id."""
        ledger = JobLedger(_record(), state_dir=tmp_path)
        first = ledger.add(_item("x"))
        second = ledger.add(_item("x", status="succeeded"))
        assert first is second
        assert first.status == "pending"

    def test_persist_recounts_progress_and_failures(self, tmp_path):
        """persist() recomputes counters; failures() lists the failed items."""
        ledger = JobLedger(_record(), state_dir=tmp_path)
        ledger.add(_item("a", status="succeeded", seq=0))
        ledger.add(_item("b", status="failed", seq=1))
        ledger.persist()
        assert ledger.progress.total == 2
        assert ledger.progress.succeeded == 1
        assert ledger.progress.failed == 1
        assert [i.item_id for i in ledger.failures()] == ["b"]

    def test_finish_partial_when_any_failure(self, tmp_path):
        """finish() selects 'partial' when a failure is present."""
        ledger = JobLedger(_record(), state_dir=tmp_path)
        ledger.add(_item("a", status="failed", seq=0))
        ledger.finish()
        assert ledger.status == "partial"

    def test_finish_succeeded_when_clean(self, tmp_path):
        """finish() selects 'succeeded' when there are no failures."""
        ledger = JobLedger(_record(), state_dir=tmp_path)
        ledger.add(_item("a", status="succeeded", seq=0))
        ledger.finish()
        assert ledger.status == "succeeded"

    def test_reload_restores_persisted_items(self, tmp_path):
        """A fresh ledger over the same job dir loads persisted item state."""
        record = _record()
        ledger = JobLedger(record, state_dir=tmp_path)
        ledger.add(_item("a", status="succeeded", seq=0))
        ledger.persist()
        reloaded = JobLedger(
            JobRecord(job_id=record.job_id, kind="ingestion", graph_id="med"),
            state_dir=tmp_path,
        )
        assert "a" in reloaded.items
        assert reloaded.items["a"].status == "succeeded"

    def test_persist_refreshes_updated_at_and_leaves_no_temp_files(self, tmp_path):
        """persist() bumps updated_at and replaces files without leftovers."""
        ledger = JobLedger(_record(), state_dir=tmp_path)
        created = ledger.record.updated_at
        ledger.persist()
        assert ledger.record.updated_at > created
        files = sorted(p.name for p in (tmp_path / "jobs" / ledger.job_id).iterdir())
        assert files == ["items.jsonl", "manifest.json"]

    def test_reload_restores_finalization_state(self, tmp_path):
        """A re-created record regains the persisted finalize flags."""
        record = _record()
        record.finalize = True
        JobLedger(record, state_dir=tmp_path).persist()
        reloaded = JobLedger(_record_with_id(record.job_id), state_dir=tmp_path)
        assert reloaded.record.finalize is True
        assert reloaded.record.finalized is False
