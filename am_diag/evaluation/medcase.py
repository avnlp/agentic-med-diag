"""MedCaseReasoning evaluation helpers."""

from __future__ import annotations

from am_diag.common.data_models import MetricResult, OpenEndedSample
from am_diag.evaluation import baml
from am_diag.evaluation.judging import judge_context


def _extract_answer_text(text: str) -> str:
    lower = text.lower()
    start = lower.rfind("<answer>")
    end = lower.rfind("</answer>")
    if start != -1 and end != -1 and end > start:
        text = text[start + len("<answer>") : end]
    return text.strip()


def _extract_reasoning_items(sample: OpenEndedSample) -> list[str]:
    """Split the stored reasoning chain into judgeable statements."""
    if not sample.reasoning_chain:
        return []
    lines = [line.strip("-• \t") for line in sample.reasoning_chain.splitlines()]
    return [line for line in lines if line]


class MedCaseReasoningEvaluator:
    """Judge diagnosis correctness and optional reasoning recall."""

    metric_name = "medcase_reasoning"

    async def evaluate(
        self,
        sample: OpenEndedSample,
        prediction: str,
    ) -> MetricResult:
        """Score a predicted diagnosis and optional reasoning coverage."""
        prediction_answer = _extract_answer_text(prediction)

        diagnosis = await baml.judge_diagnosis_correct(
            question=sample.question,
            predicted_diagnosis=prediction_answer,
            true_diagnosis=sample.reference_answer,
            context=judge_context(sample),
        )

        reasoning_items = _extract_reasoning_items(sample)
        reasoning = None
        reasoning_score = None
        if reasoning_items:
            reasoning = await baml.judge_reasoning_recall(
                ground_truth_reasons=reasoning_items,
                predicted_trace=prediction,
            )
            reasoning_score = min(max(reasoning.recall_score, 0.0), 1.0)

        details = diagnosis.model_dump()
        if reasoning is not None:
            details["reasoning_recall"] = reasoning.model_dump()
        return MetricResult(
            metric_name=self.metric_name,
            score=(
                reasoning_score
                if reasoning_score is not None
                else (1.0 if diagnosis.is_correct else 0.0)
            ),
            passed=bool(diagnosis.is_correct),
            details=details,
        )
