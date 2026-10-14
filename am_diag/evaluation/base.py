"""Evaluator contract shared across the evaluation package."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from am_diag.common.data_models import (
    MCQSample,
    MetricResult,
    OpenEndedSample,
    RARMedSample,
    RubricSample,
)


# Any sample an evaluator may receive. Each evaluator accepts its own concrete
# type; the union keeps the shared contract honest without reaching for generics.
Sample = MCQSample | OpenEndedSample | RubricSample | RARMedSample


@runtime_checkable
class Evaluator(Protocol):
    """Score one prediction for one sample as a single MetricResult."""

    metric_name: str

    async def evaluate(self, sample: Sample, prediction: str) -> MetricResult:
        """Return the metric result for ``prediction`` against ``sample``."""
        ...


__all__ = ["Evaluator", "Sample"]
