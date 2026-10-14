"""Tests for evaluation result models."""

from __future__ import annotations

from am_diag.common.data_models import MetricResult, SampleEvaluationResult


class TestSampleEvaluationResult:
    """Tests for SampleEvaluationResult."""

    def test_serialization_includes_aggregate_score(self) -> None:
        """Verify the mean score appears in dumped output."""
        result = SampleEvaluationResult(
            sample_id="s1",
            dataset="medqa",
            metrics=[
                MetricResult(metric_name="a", score=1.0, passed=True),
                MetricResult(metric_name="b", score=0.0, passed=False),
            ],
        )
        assert result.model_dump()["aggregate_score"] == 0.5

    def test_aggregate_score_is_zero_without_metrics(self) -> None:
        """Verify an empty metric list scores zero."""
        result = SampleEvaluationResult(sample_id="s1", dataset="medqa", metrics=[])
        assert result.aggregate_score == 0.0
