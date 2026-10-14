"""Tests for MedCaseReasoningEvaluator — BAML-judged diagnosis and reasoning."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from am_diag.common.data_models import OpenEndedSample
from am_diag.evaluation.medcase import MedCaseReasoningEvaluator


class TestMedCaseReasoningEvaluator:
    """Tests for MedCaseReasoningEvaluator."""

    async def test_uses_baml_diagnosis_and_reasoning(self) -> None:
        """Verify evaluator uses BAML for both diagnosis and reasoning judging."""
        sample = OpenEndedSample(
            sample_id="s1",
            question="What is the diagnosis?",
            question_stem="What is the diagnosis?",
            reference_answer="pneumonia",
            reasoning_chain="- fever suggests infection\n- cough suggests pneumonia",
            dataset="medcase_reasoning",
            split="test",
            answer="pneumonia",
            metadata={},
        )
        diagnosis = AsyncMock()
        diagnosis.return_value = type(
            "J",
            (),
            {"is_correct": True, "model_dump": lambda self: {"is_correct": True}},
        )()
        recall = AsyncMock()
        recall.return_value = type(
            "R",
            (),
            {
                "recall_score": 0.5,
                "model_dump": lambda self: {"recall_score": 0.5},
            },
        )()
        with (
            patch("am_diag.evaluation.baml.judge_diagnosis_correct", diagnosis),
            patch("am_diag.evaluation.baml.judge_reasoning_recall", recall),
        ):
            result = await MedCaseReasoningEvaluator().evaluate(sample, "pneumonia")
        assert result.passed is True
        assert result.score == 0.5
