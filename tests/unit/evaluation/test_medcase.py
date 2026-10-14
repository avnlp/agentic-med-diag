"""Tests for MedCaseReasoningEvaluator — BAML-judged diagnosis and reasoning."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from am_diag.common.data_models import OpenEndedSample
from am_diag.evaluation.medcase import MedCaseReasoningEvaluator


class _Judgement:
    """Minimal stand-in for a BAML judgement."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _sample() -> OpenEndedSample:
    return OpenEndedSample(
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

    async def test_reasoning_recall_receives_full_prediction(self) -> None:
        """Verify reasoning is judged on the whole response, not the answer tag."""
        sample = _sample()
        diagnosis = AsyncMock(return_value=_Judgement(is_correct=True))
        recall = AsyncMock(return_value=_Judgement(recall_score=1.0))
        prediction = "Fever and cough point to infection. <answer>pneumonia</answer>"
        with (
            patch("am_diag.evaluation.baml.judge_diagnosis_correct", diagnosis),
            patch("am_diag.evaluation.baml.judge_reasoning_recall", recall),
        ):
            await MedCaseReasoningEvaluator().evaluate(sample, prediction)
        assert diagnosis.await_args.kwargs["predicted_diagnosis"] == "pneumonia"
        assert recall.await_args.kwargs["predicted_trace"] == prediction

    @pytest.mark.parametrize(("raw", "expected"), [(1.7, 1.0), (-0.2, 0.0)])
    async def test_clamps_out_of_range_recall(
        self, raw: float, expected: float
    ) -> None:
        """Verify recall scores outside [0, 1] are constrained."""
        diagnosis = AsyncMock(return_value=_Judgement(is_correct=True))
        recall = AsyncMock(return_value=_Judgement(recall_score=raw))
        with (
            patch("am_diag.evaluation.baml.judge_diagnosis_correct", diagnosis),
            patch("am_diag.evaluation.baml.judge_reasoning_recall", recall),
        ):
            result = await MedCaseReasoningEvaluator().evaluate(_sample(), "pneumonia")
        assert result.score == expected
