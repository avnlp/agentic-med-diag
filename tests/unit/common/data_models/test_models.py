"""Data model tests for QA data_loaders."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from am_diag.common.data_models import (
    ConversationTurn,
    MCQSample,
    OpenEndedSample,
    RARMedSample,
    RubricCriterion,
    RubricSample,
)


def _make_mcq(**overrides) -> MCQSample:
    """Build a single-answer MCQSample from the independent fields only."""
    kwargs = {
        "sample_id": "1",
        "question": "Q?",
        "question_stem": "Q?",
        "options": {"A": "a", "B": "b"},
        "answer_keys": ["A"],
        "dataset": "test",
        "split": "test",
        "metadata": {},
    }
    kwargs.update(overrides)
    return MCQSample(**kwargs)


class TestMCQSample:
    """Tests for MCQSample data model."""

    def test_construction_minimal(self):
        """Verify minimal construction with defaults."""
        sample = _make_mcq(metadata={"k": "v"})
        assert sample.sample_id == "1"
        assert sample.options_original is None

    def test_singular_fields_are_derived(self):
        """Verify answer_key and answer_text are derived from answer_keys."""
        sample = _make_mcq(options={"A": "a", "B": "b"}, answer_keys=["B"])
        assert sample.answer_key == "B"
        assert sample.answer == "B"
        assert sample.answer == sample.answer_key
        assert sample.answer_text == "b"
        assert sample.answer_text == sample.options[sample.answer_key]

    def test_answer_texts_parallel_to_keys(self):
        """Verify answer_texts are parallel to answer_keys."""
        sample = _make_mcq(
            options={"A": "Red", "B": "Green", "C": "Blue"},
            answer_keys=["A", "C"],
            is_multi_answer=True,
        )
        assert sample.answer_texts == ["Red", "Blue"]
        assert sample.answer_key == sample.answer_keys[0]
        assert sample.answer_text == sample.answer_texts[0]
        assert sample.is_multi_answer is True

    def test_single_answer_is_multi_false_by_default(self):
        """Verify is_multi_answer defaults to False."""
        sample = _make_mcq()
        assert sample.is_multi_answer is False
        assert sample.answer_keys == ["A"]

    def test_rejects_empty_answer_keys(self):
        """Verify empty answer_keys raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_mcq(answer_keys=[])

    def test_rejects_duplicate_answer_keys(self):
        """Verify duplicate answer_keys raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_mcq(answer_keys=["A", "A"], is_multi_answer=True)

    def test_rejects_multiple_keys_when_not_multi_answer(self):
        """Verify several answer_keys without is_multi_answer raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_mcq(
                options={"A": "a", "B": "b"},
                answer_keys=["A", "B"],
                is_multi_answer=False,
            )

    def test_rejects_key_absent_from_options(self):
        """Verify key not in options raises ValidationError."""
        with pytest.raises(ValidationError):
            _make_mcq(options={"A": "a"}, answer_keys=["Z"])

    def test_options_is_dict_of_strings(self):
        """Verify options is a dict of strings."""
        sample = _make_mcq(
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            answer_keys=["C"],
        )
        assert isinstance(sample.options, dict)
        assert all(isinstance(v, str) for v in sample.options.values())

    def test_metadata_accepts_arbitrary_keys(self):
        """Verify metadata accepts arbitrary key-value pairs."""
        sample = _make_mcq(metadata={"foo": 1, "bar": [2, 3], "nested": {"x": True}})
        assert sample.metadata["foo"] == 1
        assert sample.metadata["bar"] == [2, 3]


class TestRubricSample:
    """Tests for RubricSample data model."""

    def test_construction_with_nested_types(self):
        """Verify construction with nested rubric types."""
        sample = RubricSample(
            sample_id="1",
            conversation=[ConversationTurn(role="user", content="Hi")],
            rubrics=[
                RubricCriterion(
                    criterion="test",
                    points=1,
                    tags=["axis:accuracy"],
                    axis="accuracy",
                    level=None,
                    cluster=None,
                ),
            ],
            theme="test",
            variant="all",
            criteria=["test"],
            points_list=[1],
            axes=["accuracy"],
            criterion_ids=["a" * 16],
            consensus_criteria=[None],
            metadata={},
        )
        assert len(sample.rubrics) == 1

    def test_parallel_list_lengths_match(self):
        """Verify parallel list fields have matching lengths."""
        sample = RubricSample(
            sample_id="1",
            conversation=[],
            rubrics=[],
            theme=None,
            variant="all",
            criteria=["c1", "c2"],
            points_list=[1, 2],
            axes=["a", "b"],
            criterion_ids=["x" * 16, "y" * 16],
            consensus_criteria=[None, None],
            metadata={},
        )
        assert len(sample.criteria) == len(sample.points_list)
        assert len(sample.criteria) == len(sample.axes)
        assert len(sample.criteria) == len(sample.criterion_ids)
        assert len(sample.criteria) == len(sample.consensus_criteria)

    def test_answer_defaults_to_empty_string(self):
        """Verify answer defaults to empty string."""
        sample = RubricSample(
            sample_id="1",
            conversation=[],
            rubrics=[],
            theme=None,
            variant="all",
            criteria=[],
            points_list=[],
            axes=[],
            criterion_ids=[],
            consensus_criteria=[],
            metadata={},
        )
        assert sample.answer == ""

    def test_question_defaults_to_empty_string(self):
        """Verify question defaults to empty string."""
        sample = RubricSample(
            sample_id="1",
            conversation=[],
            rubrics=[],
            theme=None,
            variant="all",
            criteria=[],
            points_list=[],
            axes=[],
            criterion_ids=[],
            consensus_criteria=[],
            metadata={},
        )
        assert sample.question == ""


class TestRARMedSample:
    """Tests for RARMedSample data model."""

    def test_answer_equals_reference_answer(self):
        """Verify answer equals reference_answer."""
        sample = RARMedSample(
            sample_id="1",
            question="Q?",
            reference_answer="A",
            rubrics=[],
            answer="A",
            metadata={},
        )
        assert sample.answer == sample.reference_answer

    def test_rubrics_is_list_of_dicts(self):
        """Verify rubrics is a list of dicts."""
        sample = RARMedSample(
            sample_id="1",
            question="Q?",
            reference_answer="A",
            rubrics=[{"criterion": "test", "points": 1}],
            answer="A",
            metadata={},
        )
        assert isinstance(sample.rubrics, list)
        assert isinstance(sample.rubrics[0], dict)


class TestOpenEndedSample:
    """Tests for OpenEndedSample data model."""

    def test_reasoning_chain_optional(self):
        """Verify reasoning_chain is optional."""
        sample = OpenEndedSample(
            sample_id="1",
            question="Q?",
            reference_answer="A",
            dataset="test",
            answer="A",
            metadata={},
        )
        assert sample.reasoning_chain is None

    def test_answer_equals_reference_answer(self):
        """Verify answer field matches reference_answer for OpenEndedSample."""
        sample = OpenEndedSample(
            sample_id="1",
            question="Q?",
            reference_answer="Diagnosis",
            reasoning_chain="Step 1",
            dataset="test",
            answer="Diagnosis",
            metadata={},
        )
        assert sample.answer == sample.reference_answer

    def test_split_defaults_to_test(self):
        """Verify split defaults to test."""
        sample = OpenEndedSample(
            sample_id="1",
            question="Q?",
            reference_answer="A",
            dataset="test",
            answer="A",
            metadata={},
        )
        assert sample.split == "test"
