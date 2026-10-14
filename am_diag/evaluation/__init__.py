"""Evaluation helpers for dataset metrics."""

from __future__ import annotations

from am_diag.common.data_models import (
    DatasetEvaluationSummary,
    MCQAccuracyResult,
    MetricResult,
    SampleEvaluationResult,
)
from am_diag.evaluation.baml import (
    judge_answer_equivalent,
    judge_diagnosis_correct,
    judge_reasoning_recall,
    judge_rubric_criterion,
)
from am_diag.evaluation.base import Evaluator, Sample
from am_diag.evaluation.diagnostic import DiagnosisAccuracyEvaluator
from am_diag.evaluation.judging import binary_metric, judge_context
from am_diag.evaluation.mcq import MCQAccuracyEvaluator
from am_diag.evaluation.medcase import MedCaseReasoningEvaluator
from am_diag.evaluation.open_ended import AnswerEquivalenceEvaluator
from am_diag.evaluation.rubric import RubricEvaluator


__all__ = [
    "AnswerEquivalenceEvaluator",
    "DatasetEvaluationSummary",
    "DiagnosisAccuracyEvaluator",
    "Evaluator",
    "MCQAccuracyEvaluator",
    "MCQAccuracyResult",
    "MedCaseReasoningEvaluator",
    "MetricResult",
    "RubricEvaluator",
    "Sample",
    "SampleEvaluationResult",
    "binary_metric",
    "judge_answer_equivalent",
    "judge_context",
    "judge_diagnosis_correct",
    "judge_reasoning_recall",
    "judge_rubric_criterion",
]
