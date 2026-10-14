"""Tests for judging helpers — judge_context and binary_metric."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from am_diag.evaluation.judging import binary_metric, judge_context


class _Judgement:
    def __init__(self, *, is_correct: bool) -> None:
        self.is_correct = is_correct

    def model_dump(self) -> dict[str, Any]:
        return {"is_correct": self.is_correct}


class TestJudgeContext:
    """Tests for judge_context helper."""

    def test_prefers_for_judge_key(self) -> None:
        """Verify retrieved_context_for_judge is preferred."""
        sample = SimpleNamespace(
            metadata={
                "retrieved_context_for_judge": "judge",
                "retrieved_context": "plain",
            },
        )
        assert judge_context(sample) == "judge"

    def test_falls_back_to_retrieved_context(self) -> None:
        """Verify fallback to retrieved_context."""
        sample = SimpleNamespace(metadata={"retrieved_context": "plain"})
        assert judge_context(sample) == "plain"

    def test_none_when_neither_key_present(self) -> None:
        """Verify None returned when neither key present."""
        sample = SimpleNamespace(metadata={"other": "x"})
        assert judge_context(sample) is None

    def test_none_when_metadata_empty(self) -> None:
        """Verify None returned when metadata is empty."""
        assert judge_context(SimpleNamespace(metadata={})) is None
        assert judge_context(SimpleNamespace(metadata=None)) is None


class TestBinaryMetric:
    """Tests for binary_metric helper."""

    def test_correct_judgement(self) -> None:
        """Verify correct judgement returns score 1.0."""
        result = binary_metric("answer_equivalence", _Judgement(is_correct=True))
        assert result.metric_name == "answer_equivalence"
        assert result.score == 1.0
        assert result.passed is True
        assert result.details == {"is_correct": True}

    def test_incorrect_judgement(self) -> None:
        """Verify incorrect judgement returns score 0.0."""
        result = binary_metric("answer_equivalence", _Judgement(is_correct=False))
        assert result.score == 0.0
        assert result.passed is False
        assert result.details == {"is_correct": False}
