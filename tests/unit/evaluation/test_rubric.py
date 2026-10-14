"""Tests for RubricEvaluator — rubric-scored evaluation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from am_diag.common.data_models import RubricSample
from am_diag.evaluation.rubric import RubricEvaluator


class TestRubricEvaluator:
    """Tests for RubricEvaluator."""

    async def test_empty_rubrics_scores_zero(self) -> None:
        """Verify empty rubrics returns score 0."""
        sample = RubricSample(
            sample_id="s1",
            conversation=[],
            rubrics=[],
            theme=None,
            variant="v1",
            criteria=[],
            points_list=[],
            axes=[],
            criterion_ids=[],
            consensus_criteria=[],
            metadata={},
        )
        fake = AsyncMock()
        with patch("am_diag.evaluation.baml.judge_rubric_criterion", fake):
            result = await RubricEvaluator().evaluate(sample, "resp")
        assert result.score == 0.0
        assert result.metric_name == "rubric_score"
