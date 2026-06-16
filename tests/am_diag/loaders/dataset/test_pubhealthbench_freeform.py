"""Unit tests for PubHealthBench freeform (open-ended) dataset loader."""

from __future__ import annotations

from am_diag.loaders.dataset.pubhealthbench_freeform import (
    PubHealthBenchFreeformDataset,
)


class TestPubHealthBenchFreeformDataset:
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    def test_empty_question_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [self._make_row(question="")],
        )
        samples = PubHealthBenchFreeformDataset().load()
        assert len(samples) == 0

    def test_reference_answer_from_options_and_index(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [self._make_row()],
        )
        samples = PubHealthBenchFreeformDataset().load()
        assert samples[0].reference_answer == "Statement B"

    def test_reference_answer_fallback_when_no_options(self, patch_load_dataset):
        row = self._make_row(options=[], answer_index=None)
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [row],
        )
        samples = PubHealthBenchFreeformDataset().load()
        assert samples[0].reference_answer == ""

    def test_retrieved_context_for_judge_in_metadata(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [self._make_row(retrieved_context_for_judge="Context text")],
        )
        samples = PubHealthBenchFreeformDataset().load()
        assert "retrieved_context_for_judge" in samples[0].metadata
        assert samples[0].metadata["retrieved_context_for_judge"] == "Context text"

    def test_missing_retrieved_context_gets_empty_string(self, patch_load_dataset):
        row = self._make_row(retrieved_context_for_judge=None)
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [row],
        )
        samples = PubHealthBenchFreeformDataset().load()
        assert "retrieved_context_for_judge" in samples[0].metadata
        assert samples[0].metadata["retrieved_context_for_judge"] == ""

    def test_default_split_is_test(self):
        loader = PubHealthBenchFreeformDataset()
        assert loader.default_split == "test"

    def test_limit_param(self, patch_load_dataset):
        rows = [self._make_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.base", rows)
        samples = PubHealthBenchFreeformDataset().load(limit=5)
        assert len(samples) <= 5

    def _make_row(self, **overrides):
        row = {
            "question": "What is the correct statement?",
            "options": ["Statement A", "Statement B", "Statement C"],
            "answer_index": 1,
            "question_id": "phb_001",
            "category": "Epidemiology",
            "source": "CDC",
            "retrieved_context_for_judge": "Context text",
        }
        row.update(overrides)
        return row
