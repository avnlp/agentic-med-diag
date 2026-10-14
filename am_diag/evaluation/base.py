"""Evaluator contract shared across the evaluation package."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from am_diag.common.data_models import (
    MCQSample,
    MetricResult,
    OpenEndedSample,
    RARMedSample,
    RubricSample,
)


Sample = MCQSample | OpenEndedSample | RubricSample | RARMedSample

SampleT_contra = TypeVar("SampleT_contra", contravariant=True)


@runtime_checkable
class Evaluator(Protocol[SampleT_contra]):
    """Score one prediction for one sample as a single MetricResult.

    Each evaluator accepts its own sample type, so the protocol is generic
    over it.
    """

    metric_name: str

    async def evaluate(
        self,
        sample: SampleT_contra,
        prediction: str,
    ) -> MetricResult:
        """Return the metric result for ``prediction`` against ``sample``."""
        ...


__all__ = ["Evaluator", "Sample"]
