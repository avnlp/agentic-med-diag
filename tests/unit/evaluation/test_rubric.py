"""Tests for RubricEvaluator — rubric-scored evaluation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from am_diag.common.data_models import (
    ConversationTurn,
    RubricCriterion,
    RubricSample,
)
from am_diag.evaluation.rubric import RubricEvaluator


def _sample(points: list[int], *, conversation: bool = False) -> RubricSample:
    rubrics = [
        RubricCriterion(
            criterion=f"criterion {index}",
            points=value,
            tags=[],
            axis=None,
            level=None,
            cluster=None,
        )
        for index, value in enumerate(points)
    ]
    turns = (
        [
            ConversationTurn(role="user", content="I have a fever."),
            ConversationTurn(role="assistant", content="For how long?"),
            ConversationTurn(role="user", content="Two days."),
        ]
        if conversation
        else []
    )
    return RubricSample(
        sample_id="s1",
        conversation=turns,
        rubrics=rubrics,
        theme=None,
        variant="v1",
        criteria=[r.criterion for r in rubrics],
        points_list=points,
        axes=[],
        criterion_ids=[f"id-{index}" for index in range(len(points))],
        consensus_criteria=[],
        metadata={},
    )


def _judge(*met: bool) -> AsyncMock:
    return AsyncMock(
        side_effect=[
            SimpleNamespace(criteria_met=value, model_dump=lambda: {}) for value in met
        ],
    )


class TestRubricEvaluator:
    """Tests for RubricEvaluator."""

    async def test_empty_rubrics_scores_zero(self) -> None:
        """Verify empty rubrics returns score 0 without calling the judge."""
        fake = AsyncMock()
        with patch("am_diag.evaluation.baml.judge_rubric_criterion", fake):
            result = await RubricEvaluator().evaluate(_sample([]), "resp")
        fake.assert_not_awaited()
        assert result.score == 0.0
        assert result.metric_name == "rubric_score"

    @pytest.mark.parametrize(
        ("points", "met", "expected"),
        [
            ([5, 3], (True, False), 5 / 8),
            ([5, -4], (True, True), 0.2),
            ([5, -4], (False, True), 0.0),
            ([5, -4], (True, False), 1.0),
            ([-4], (True,), 0.0),
        ],
    )
    async def test_scores_signed_points_within_unit_range(
        self,
        points: list[int],
        met: tuple[bool, ...],
        expected: float,
    ) -> None:
        """Verify negative criteria subtract and the score stays in [0, 1]."""
        with patch(
            "am_diag.evaluation.baml.judge_rubric_criterion",
            _judge(*met),
        ):
            result = await RubricEvaluator().evaluate(_sample(points), "resp")
        assert result.score == pytest.approx(expected)

    async def test_forwards_conversation_and_criterion_ids(self) -> None:
        """Verify the judge sees the full conversation and each criterion's ID."""
        fake = _judge(True, False)
        with patch("am_diag.evaluation.baml.judge_rubric_criterion", fake):
            await RubricEvaluator().evaluate(_sample([2, 1], conversation=True), "resp")
        first, second = (call.kwargs for call in fake.await_args_list)
        assert first["question_or_conversation"] == (
            "user: I have a fever.\nassistant: For how long?\nuser: Two days."
        )
        assert (first["criterion_id"], second["criterion_id"]) == ("id-0", "id-1")
