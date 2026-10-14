"""BAML judge adapters for evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from am_diag.llm.baml_client.types import (
        AnswerEquivalenceJudgement,
        DiagnosisJudgement,
        ReasoningRecallJudgement,
        RubricCriterionJudgement,
    )


async def judge_answer_equivalent(
    question: str,
    predicted_answer: str,
    reference_answer: str,
    context: str | None = None,
) -> AnswerEquivalenceJudgement:
    """Ask the BAML judge whether an answer is equivalent."""
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    return await b.JudgeAnswerEquivalent(
        question=question,
        predicted_answer=predicted_answer,
        reference_answer=reference_answer,
        context=context,
    )


async def judge_diagnosis_correct(
    question: str,
    predicted_diagnosis: str,
    true_diagnosis: str,
    context: str | None = None,
) -> DiagnosisJudgement:
    """Ask the BAML judge whether a diagnosis matches."""
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    return await b.JudgeDiagnosisCorrect(
        question=question,
        predicted_diagnosis=predicted_diagnosis,
        true_diagnosis=true_diagnosis,
        context=context,
    )


async def judge_rubric_criterion(
    question_or_conversation: str,
    assistant_response: str,
    criterion_text: str,
    criterion_id: str | None = None,
    points: int | None = None,
) -> RubricCriterionJudgement:
    """Ask the BAML judge whether a rubric criterion is met."""
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    return await b.JudgeRubricCriterion(
        question_or_conversation=question_or_conversation,
        assistant_response=assistant_response,
        criterion_text=criterion_text,
        criterion_id=criterion_id,
        points=points,
    )


async def judge_reasoning_recall(
    ground_truth_reasons: list[str],
    predicted_trace: str,
) -> ReasoningRecallJudgement:
    """Ask the BAML judge for reasoning recall coverage."""
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    return await b.JudgeReasoningRecall(
        ground_truth_reasons=ground_truth_reasons,
        predicted_trace=predicted_trace,
    )
