"""Tests for MCQAccuracyEvaluator — extracts letter from answer tag."""

from __future__ import annotations

from am_diag.common.data_models import MCQSample
from am_diag.evaluation.mcq import MCQAccuracyEvaluator


class TestMCQAccuracyEvaluator:
    """Tests for MCQAccuracyEvaluator."""

    async def test_extracts_letter_from_answer_tag(self) -> None:
        """Verify letter is extracted from answer XML tag."""
        sample = MCQSample(
            sample_id="s1",
            question="Q",
            question_stem="Q",
            options={"A": "yes", "B": "no"},
            answer_keys=["A"],
            dataset="medqa",
            split="test",
            metadata={},
        )
        result = await MCQAccuracyEvaluator().evaluate(sample, "<answer>A</answer>")
        assert result.passed is True
        assert result.score == 1.0
