"""Tests for MCQAccuracyEvaluator — extracts letter from answer tag."""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.evaluation.mcq import MCQAccuracyEvaluator


def _sample(options: dict[str, str], answer_keys: list[str]) -> MCQSample:
    return MCQSample(
        sample_id="s1",
        question="Q",
        question_stem="Q",
        options=options,
        answer_keys=answer_keys,
        dataset="medqa",
        split="test",
        metadata={},
    )


class TestMCQAccuracyEvaluator:
    """Tests for MCQAccuracyEvaluator."""

    async def test_extracts_letter_from_answer_tag(self) -> None:
        """Verify letter is extracted from answer XML tag."""
        sample = MCQSample(
            sample_id="s1",
            question="Q",
            question_stem="Q",
            options={"A": "yes", "B": "no"},
            answer_keys=["A"],
            dataset="medqa",
            split="test",
            metadata={},
        )
        result = await MCQAccuracyEvaluator().evaluate(sample, "<answer>A</answer>")
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.parametrize(
        ("prediction", "expected"),
        [
            ("b", True),
            ("Answer: B", True),
            ("<answer>B</answer>", True),
            ("The answer is B.", True),
            ("Option A is tempting, but the answer is B", True),
            ("The answer is A", False),
        ],
    )
    async def test_lowercase_answer_keys(self, prediction: str, expected: bool) -> None:
        """Verify lowercase option labels are graded like uppercase ones."""
        sample = _sample({"a": "yes", "b": "no"}, ["b"])
        result = await MCQAccuracyEvaluator().evaluate(sample, prediction)
        assert result.passed is expected

    @pytest.mark.parametrize(
        "prediction",
        [
            "This is a case of a viral infection",
            "I would say it is not B because of the rash, which is a sign of a",
            "Patients with this finding are usually given B-blockers a week later",
        ],
    )
    async def test_prose_without_explicit_answer_does_not_match(
        self,
        prediction: str,
    ) -> None:
        """Verify stray capital letters or articles in prose are not answers."""
        sample = _sample({"A": "yes", "B": "no"}, ["A"])
        result = await MCQAccuracyEvaluator().evaluate(sample, prediction)
        assert result.passed is False
        assert result.details["method"] == "no_match"

    async def test_multi_answer_with_lowercase_keys(self) -> None:
        """Verify multi-answer sets compare case-insensitively."""
        sample = MCQSample(
            sample_id="s1",
            question="Q",
            question_stem="Q",
            options={"a": "x", "b": "y", "c": "z"},
            answer_keys=["a", "c"],
            is_multi_answer=True,
            dataset="medqa",
            split="test",
            metadata={},
        )
        result = await MCQAccuracyEvaluator().evaluate(sample, "<answer>A, C</answer>")
        assert result.passed is True
