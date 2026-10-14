"""Open-ended answer-equivalence evaluator."""

from __future__ import annotations

from am_diag.common.data_models import MetricResult, OpenEndedSample
from am_diag.evaluation import baml
from am_diag.evaluation.judging import binary_metric, judge_context


class AnswerEquivalenceEvaluator:
    """Use BAML to judge generic answer equivalence."""

    metric_name = "answer_equivalence"

    async def evaluate(self, sample: OpenEndedSample, prediction: str) -> MetricResult:
        """Judge whether the prediction matches the reference answer."""
        judgement = await baml.judge_answer_equivalent(
            question=sample.question,
            predicted_answer=prediction,
            reference_answer=sample.reference_answer,
            context=judge_context(sample),
        )
        return binary_metric(self.metric_name, judgement)
