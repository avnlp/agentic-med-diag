"""Data model tests for QA data_loaders."""

from __future__ import annotations

from am_diag.common.data_models import (
    ConversationTurn,
    MCQSample,
    OpenEndedSample,
    RARMedSample,
    RubricCriterion,
    RubricSample,
)


class TestMCQSample:
    def test_construction_with_all_fields(self):
        sample = MCQSample(
            sample_id="1",
            question="Q?",
            question_stem="Q?",
            options={"A": "a", "B": "b"},
            answer_key="A",
            answer_text="a",
            answer="A",
            answer_keys=["A"],
            answer_texts=["a"],
            dataset="test",
            split="test",
            metadata={"k": "v"},
        )
        assert sample.sample_id == "1"

    def test_answer_equals_answer_key_invariant(self):
        sample = MCQSample(
            sample_id="1",
            question="Q?",
            question_stem="Q?",
            options={"A": "a", "B": "b"},
            answer_key="B",
            answer_text="b",
            answer="B",
            answer_keys=["B"],
            answer_texts=["b"],
            dataset="test",
            split="test",
            metadata={},
        )
        assert sample.answer == sample.answer_key

    def test_answer_text_equals_options_lookup(self):
        sample = MCQSample(
            sample_id="1",
            question="Q?",
            question_stem="Q?",
            options={"A": "alpha", "B": "beta"},
            answer_key="A",
            answer_text="alpha",
            answer="A",
            answer_keys=["A"],
            answer_texts=["alpha"],
            dataset="test",
            split="test",
            metadata={},
        )
        assert sample.answer_text == sample.options[sample.answer_key]

    def test_options_is_dict_of_strings(self):
        sample = MCQSample(
            sample_id="1",
            question="Q?",
            question_stem="Q?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            answer_key="C",
            answer_text="c",
            answer="C",
            answer_keys=["C"],
            answer_texts=["c"],
            dataset="test",
            split="test",
            metadata={},
        )
        assert isinstance(sample.options, dict)
        assert all(isinstance(v, str) for v in sample.options.values())

    def test_metadata_accepts_arbitrary_keys(self):
        sample = MCQSample(
            sample_id="1",
            question="Q?",
            question_stem="Q?",
            options={"A": "a"},
            answer_key="A",
            answer_text="a",
            answer="A",
            answer_keys=["A"],
            answer_texts=["a"],
            dataset="test",
            split="test",
            metadata={"foo": 1, "bar": [2, 3], "nested": {"x": True}},
        )
        assert sample.metadata["foo"] == 1
        assert sample.metadata["bar"] == [2, 3]

    def test_multi_answer_fields_populated(self):
        sample = MCQSample(
            sample_id="1",
            question="Q?",
            question_stem="Q?",
            options={"A": "Red", "B": "Green", "C": "Blue"},
            answer_key="A",
            answer_text="Red",
            answer="A",
            answer_keys=["A", "C"],
            answer_texts=["Red", "Blue"],
            is_multi_answer=True,
            dataset="test",
            split="test",
            metadata={},
        )
        assert sample.answer_keys == ["A", "C"]
        assert sample.answer_texts == ["Red", "Blue"]
        assert sample.is_multi_answer is True
        # Backward compat invariants still hold
        assert sample.answer == sample.answer_key
        assert sample.answer_text == sample.options[sample.answer_key]
        assert sample.answer_key == sample.answer_keys[0]
        assert sample.answer_text == sample.answer_texts[0]

    def test_single_answer_defaults(self):
        sample = MCQSample(
            sample_id="1",
            question="Q?",
            question_stem="Q?",
            options={"A": "x"},
            answer_key="A",
            answer_text="x",
            answer="A",
            dataset="test",
            split="test",
            metadata={},
        )
        assert sample.answer_keys == []
        assert sample.answer_texts == []
        assert sample.is_multi_answer is False


class TestRubricSample:
    def test_construction_with_nested_types(self):
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
    def test_answer_equals_reference_answer(self):
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
    def test_reasoning_chain_optional(self):
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
        sample = OpenEndedSample(
            sample_id="1",
            question="Q?",
            reference_answer="A",
            dataset="test",
            answer="A",
            metadata={},
        )
        assert sample.split == "test"
