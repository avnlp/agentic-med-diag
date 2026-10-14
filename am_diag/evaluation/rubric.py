"""Rubric scoring helpers."""

from __future__ import annotations

from pydantic import BaseModel

from am_diag.common.data_models import MetricResult, RARMedSample, RubricSample
from am_diag.evaluation import baml


class RubricEvaluator:
    """Use BAML to judge rubric criteria (weighted points)."""

    metric_name = "rubric_score"

    async def evaluate(
        self,
        sample: RubricSample | RARMedSample,
        prediction: str,
    ) -> MetricResult:
        """Score a prediction using rubric-level judgments."""
        criteria = []
        total_points = 0
        met_points = 0
        for rubric in sample.rubrics:
            if isinstance(rubric, BaseModel):
                rubric_data = rubric.model_dump()
            else:
                rubric_data = dict(rubric)
            criterion_text = rubric_data.get("criterion", "")
            criterion_id = rubric_data.get("criterion_id")
            points = rubric_data.get("points", 0)
            total_points += points
            judgement = await baml.judge_rubric_criterion(
                question_or_conversation=sample.question,
                assistant_response=prediction,
                criterion_text=criterion_text,
                criterion_id=criterion_id,
                points=points,
            )
            met = bool(judgement.criteria_met)
            if met:
                met_points += points
            criteria.append({"points": points, "judgement": judgement.model_dump()})
        score = met_points / total_points if total_points else 0.0
        return MetricResult(
            metric_name=self.metric_name,
            score=score,
            passed=bool(total_points) and score >= 0.5,
            details={"criteria": criteria, "total_points": total_points},
        )
