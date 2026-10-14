"""Tests for AnswerEquivalenceEvaluator — BAML-judged answer equivalence."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from am_diag.common.data_models import OpenEndedSample
from am_diag.evaluation.open_ended import AnswerEquivalenceEvaluator


class TestAnswerEquivalenceEvaluator:
    """Tests for AnswerEquivalenceEvaluator."""

    async def test_exact_match(self) -> None:
        """Verify exact match passes evaluation."""
        sample = OpenEndedSample(
            sample_id="s1",
            question="Q",
            question_stem="Q",
            reference_answer="diabetes",
            reasoning_chain=None,
            dataset="nejm_diagnostic",
            split="test",
            answer="diabetes",
            metadata={},
        )
        fake = AsyncMock()
        fake.return_value = type(
            "J",
            (),
            {"is_correct": True, "model_dump": lambda self: {"is_correct": True}},
        )()
        with patch("am_diag.evaluation.baml.judge_answer_equivalent", fake):
            result = await AnswerEquivalenceEvaluator().evaluate(sample, "Diabetes")
        assert result.passed is True
