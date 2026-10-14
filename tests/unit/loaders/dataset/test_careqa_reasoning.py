"""Unit tests for CareQA open-ended reasoning dataset loader."""

from __future__ import annotations

from am_diag.loaders.dataset.careqa_reasoning import CareQAReasoningDataset


class TestCareQAReasoningDataset:
    """Tests for CareQA reasoning fallback and field mapping."""

    def _make_row(self, **overrides):
        row = {
            "question": "What is the treatment?",
            "answer": "Drug A",
            "answer_explanation": "Drug A is first-line because...",
            "subject": "Cardiology",
        }
        row.update(overrides)
        return row

    def test_empty_question_returns_none(self, patch_load_dataset):
        """Verify row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [self._make_row(question="")],
        )
        samples = CareQAReasoningDataset().load()
        assert len(samples) == 0

    def test_answer_explanation_preferred_over_answer(self, patch_load_dataset):
        """Verify reference_answer prefers answer_explanation when both are present."""
        row = self._make_row(
            answer_explanation="Explanation text",
            answer="Drug A",
        )
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = CareQAReasoningDataset().load()
        assert samples[0].reference_answer == "Explanation text"

    def test_falls_back_to_answer_when_no_explanation(self, patch_load_dataset):
        """Verify reference_answer falls back to answer when explanation is None."""
        row = self._make_row(answer_explanation=None, answer="Drug A")
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = CareQAReasoningDataset().load()
        assert samples[0].reference_answer == "Drug A"

    def test_answer_equals_reference_answer(self, patch_load_dataset):
        """Verify answer and reference_answer are always equal on loaded samples."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [self._make_row()],
        )
        samples = CareQAReasoningDataset().load()
        assert samples[0].answer == samples[0].reference_answer

    def test_uses_careqa_en_open_config(self, patch_load_dataset):
        """Verify load runs without error using the careqa_en_open config."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [self._make_row()],
        )
        samples = CareQAReasoningDataset().load()
        assert len(samples) == 1
