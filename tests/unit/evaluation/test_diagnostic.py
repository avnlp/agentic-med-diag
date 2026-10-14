"""Tests for DiagnosisAccuracyEvaluator — BAML-judged diagnosis accuracy."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from am_diag.common.data_models import OpenEndedSample
from am_diag.evaluation.diagnostic import DiagnosisAccuracyEvaluator


class TestDiagnosisAccuracyEvaluator:
    """Tests for DiagnosisAccuracyEvaluator."""

    async def test_uses_baml(self) -> None:
        """Verify evaluator uses BAML for judging."""
        sample = OpenEndedSample(
            sample_id="s2",
            question="Q2",
            question_stem="Q2",
            reference_answer="pneumonia",
            reasoning_chain=None,
            dataset="nejm_diagnostic",
            split="test",
            answer="pneumonia",
            metadata={},
        )
        fake = AsyncMock()
        fake.return_value = type(
            "J",
            (),
            {"is_correct": True, "model_dump": lambda self: {"is_correct": True}},
        )()
        with patch("am_diag.evaluation.baml.judge_diagnosis_correct", fake):
            result = await DiagnosisAccuracyEvaluator().evaluate(sample, "pneumonia")
        assert result.passed is True
        assert result.metric_name == "diagnosis_accuracy"
