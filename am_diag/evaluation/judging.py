"""Pure helpers shared by the BAML-judge evaluators.

Both functions are LLM-free leaf helpers: they hold the two pieces of logic the
judge evaluators would otherwise copy verbatim. Keeping them here means the
invariants live (and are tested) in one place, and a debugger steps straight
through them rather than through an indirection layer.
"""

from __future__ import annotations

from typing import Any, Protocol

from am_diag.common.data_models import MetricResult


class _BinaryJudgement(Protocol):
    """Shape of a BAML judgement scored as correct/incorrect."""

    is_correct: bool

    def model_dump(self) -> dict[str, Any]: ...


def judge_context(sample: Any) -> str | None:
    """Pull the judge-facing retrieved context from a sample's metadata.

    Prefers ``retrieved_context_for_judge``, falls back to ``retrieved_context``,
    and returns ``None`` when neither (or no metadata) is present.
    """
    metadata = getattr(sample, "metadata", None)
    if not metadata:
        return None
    context = metadata.get("retrieved_context_for_judge")
    if context is None:
        context = metadata.get("retrieved_context")
    return context


def binary_metric(metric_name: str, judgement: _BinaryJudgement) -> MetricResult:
    """Wrap a correct/incorrect BAML judgement as a MetricResult."""
    correct = bool(judgement.is_correct)
    return MetricResult(
        metric_name=metric_name,
        score=1.0 if correct else 0.0,
        passed=correct,
        details=judgement.model_dump(),
    )


__all__ = ["binary_metric", "judge_context"]
