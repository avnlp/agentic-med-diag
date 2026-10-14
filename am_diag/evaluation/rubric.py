"""Rubric scoring helpers."""

from __future__ import annotations

from pydantic import BaseModel

from am_diag.common.data_models import MetricResult, RARMedSample, RubricSample
from am_diag.evaluation import baml


def _conversation_text(sample: RubricSample | RARMedSample) -> str:
    """Return the task or multi-turn conversation shown to the judge."""
    if isinstance(sample, RubricSample) and sample.conversation:
        return "\n".join(f"{turn.role}: {turn.content}" for turn in sample.conversation)
    return sample.question


class RubricEvaluator:
    """Use BAML to judge rubric criteria (weighted points).

    The score follows the HealthBench rule: points of the met criteria, which
    may be negative, divided by the sum of the positive points and clipped to
    ``[0, 1]``.
    """

    metric_name = "rubric_score"

    async def evaluate(
        self,
        sample: RubricSample | RARMedSample,
        prediction: str,
    ) -> MetricResult:
        """Score a prediction using rubric-level judgments."""
        conversation = _conversation_text(sample)
        criterion_ids = sample.criterion_ids if isinstance(sample, RubricSample) else []
        criteria = []
        positive_points = 0
        earned_points = 0
        for index, rubric in enumerate(sample.rubrics):
            if isinstance(rubric, BaseModel):
                rubric_data = rubric.model_dump()
            else:
                rubric_data = dict(rubric)
            criterion_text = rubric_data.get("criterion", "")
            criterion_id = criterion_ids[index] if index < len(criterion_ids) else None
            points = rubric_data.get("points", 0)
            positive_points += max(points, 0)
            judgement = await baml.judge_rubric_criterion(
                question_or_conversation=conversation,
                assistant_response=prediction,
                criterion_text=criterion_text,
                criterion_id=criterion_id,
                points=points,
            )
            if judgement.criteria_met:
                earned_points += points
            criteria.append({"points": points, "judgement": judgement.model_dump()})
        score = (
            min(max(earned_points / positive_points, 0.0), 1.0)
            if positive_points
            else 0.0
        )
        return MetricResult(
            metric_name=self.metric_name,
            score=score,
            passed=score >= 0.5,
            details={
                "criteria": criteria,
                "total_points": positive_points,
                "earned_points": earned_points,
            },
        )
