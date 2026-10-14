"""Diagnosis-accuracy evaluator."""

from __future__ import annotations

from am_diag.common.data_models import MetricResult, OpenEndedSample
from am_diag.evaluation import baml
from am_diag.evaluation.judging import binary_metric, judge_context


class DiagnosisAccuracyEvaluator:
    """Use BAML to judge diagnosis equivalence."""

    metric_name = "diagnosis_accuracy"

    async def evaluate(self, sample: OpenEndedSample, prediction: str) -> MetricResult:
        """Score a diagnosis prediction against the reference answer."""
        judgement = await baml.judge_diagnosis_correct(
            question=sample.question,
            predicted_diagnosis=prediction,
            true_diagnosis=sample.reference_answer,
            context=judge_context(sample),
        )
        return binary_metric(self.metric_name, judgement)
