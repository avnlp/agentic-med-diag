"""Unit tests for NEJM AI Q&A MCQ dataset loader."""

from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.nejm_qa import NEJMQADataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_mcq_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question": "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid",
        "answer": "B",
    }
    row.update(overrides)
    return row


class TestNEJMQADataset(ShuffleTestMixin):
    """Order: Edge/empty -> Core behavior -> Multi-answer -> Fallback -> Fixtures."""

    loader_class = NEJMQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_mcq_row(**overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset([self.make_row()])
        return self.loader_class(**loader_kwargs).load(split="pediatrics")

    def test_skips_row_with_empty_question(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row(question="")])
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_empty_answer(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row(answer="")])
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer_letter(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row(answer="Z")])
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_skips_row_without_inline_options(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row(question="Just a plain question?")])
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_load_returns_correct_type(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset, patch_single_specialty):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset, patch_single_specialty):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_parsed_stem_and_options(self, patch_load_dataset, patch_single_specialty):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert len(samples) == 1
        assert samples[0].question_stem == "What is the capital of France?"
        assert samples[0].options == {
            "A": "London",
            "B": "Paris",
            "C": "Berlin",
            "D": "Madrid",
        }
        assert samples[0].answer_key == "B"
        assert samples[0].answer_text == "Paris"

    def test_dataset_field_correct(self, patch_load_dataset, patch_single_specialty):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert samples[0].dataset == "nejm_qa"

    def test_split_field_reflects_specialty(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert samples[0].split == "pediatrics"

    def test_sample_id_includes_specialty(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert samples[0].sample_id == "pediatrics_0"

    def test_specialty_in_metadata(self, patch_load_dataset, patch_single_specialty):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert samples[0].metadata["specialty"] == "pediatrics"

    def test_answer_raw_in_metadata(self, patch_load_dataset, patch_single_specialty):
        patch_load_dataset([make_mcq_row(answer="C")])
        samples = NEJMQADataset().load()
        assert samples[0].metadata["answer_raw"] == "C"
        assert samples[0].metadata["answer_letters"] == ["C"]

    def test_multi_answer_populates_all_fields(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        row = make_mcq_row(
            question="Which are primary colors?\nA. Red\nB. Green\nC. Blue\nD. Yellow",
            answer="A,C",
        )
        patch_load_dataset([row])
        samples = NEJMQADataset().load()
        assert len(samples) == 1
        s = samples[0]
        assert s.is_multi_answer is True
        assert s.answer_keys == ["A", "C"]
        assert s.answer_texts == ["Red", "Blue"]
        # Backward compat
        assert s.answer_key == "A"
        assert s.answer == "A"
        assert s.answer_text == "Red"
        # Invariants
        assert s.answer == s.answer_key
        assert s.answer_text == s.options[s.answer_key]
        assert s.answer_key == s.answer_keys[0]
        assert s.answer_text == s.answer_texts[0]
        assert all(
            s.answer_texts[i] == s.options[k] for i, k in enumerate(s.answer_keys)
        )
        # Metadata
        assert s.metadata["answer_raw"] == "A,C"
        assert s.metadata["answer_letters"] == ["A", "C"]

    def test_multi_answer_prompt_uses_plural_instruction(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        row = make_mcq_row(
            question="Which are primary colors?\nA. Red\nB. Green\nC. Blue\nD. Yellow",
            answer="A,C",
        )
        patch_load_dataset([row])
        samples = NEJMQADataset().load()
        assert "letter(s) of the correct option(s)" in samples[0].question

    def test_single_answer_prompt_uses_singular_instruction(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        patch_load_dataset([make_mcq_row()])
        samples = NEJMQADataset().load()
        assert "only the letter" in samples[0].question
        assert "letter(s)" not in samples[0].question

    def test_multi_answer_skips_when_no_valid_letter(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        row = make_mcq_row(
            question="Which?\nA. One\nB. Two",
            answer="X,Y",
        )
        patch_load_dataset([row])
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_multi_answer_shuffle_tracks_all_keys(
        self,
        patch_load_dataset,
    ):
        row = make_mcq_row(
            question="Which are primary colors?\nA. Red\nB. Green\nC. Blue\nD. Yellow",
            answer="A,C",
        )
        patch_load_dataset([row])
        samples = NEJMQADataset(shuffle_options=True, shuffle_seed=42).load()
        s = samples[0]
        assert s.is_multi_answer is True
        assert len(s.answer_keys) == 2
        # All answer letters still point to valid options
        for k in s.answer_keys:
            assert k in s.options
        for t in s.answer_texts:
            assert t in s.options.values()
        # Invariants hold after shuffle
        assert s.answer == s.answer_key
        assert s.answer_key == s.answer_keys[0]
        assert s.answer_text == s.answer_texts[0]
        assert all(
            s.answer_texts[i] == s.options[k] for i, k in enumerate(s.answer_keys)
        )

    def test_limit_param(self, patch_load_dataset, patch_single_specialty):
        rows = [make_mcq_row() for _ in range(10)]
        patch_load_dataset(rows)
        samples = NEJMQADataset().load(limit=3)
        assert len(samples) <= 3

    def test_load_specific_specialty(self, patch_load_dataset):
        rows = [make_mcq_row()]
        patch_load_dataset(rows)
        samples = NEJMQADataset().load(split="pediatrics")
        assert len(samples) == 1
        assert samples[0].split == "pediatrics"

    def test_unknown_split_raises(self):
        with pytest.raises(ValueError, match="Unknown split"):
            NEJMQADataset().load(split="nonexistent")

    def test_extract_stem_and_options_parses_correctly(self):
        ds = NEJMQADataset()
        text = "What is 2+2?\nA. 3\nB. 4\nC. 5"
        result = ds._extract_stem_and_options(text)
        assert result is not None
        stem, options = result
        assert stem == "What is 2+2?"
        assert options == {"A": "3", "B": "4", "C": "5"}

    def test_extract_stem_and_options_returns_none_for_plain_text(self):
        ds = NEJMQADataset()
        assert ds._extract_stem_and_options("Just a plain question.") is None

    def test_extract_stem_and_options_requires_min_two_options(self):
        ds = NEJMQADataset()
        text = "Test?\nA. Only one"
        assert ds._extract_stem_and_options(text) is None

    @pytest.fixture(autouse=True)
    def patch_single_specialty(self, mocker):
        mocker.patch("am_diag.loaders.dataset.nejm_qa._SPECIALTIES", ["pediatrics"])

    @pytest.fixture
    def make_iterable_dataset(self):
        def _make(rows: list[dict[str, Any]]):
            return Dataset.from_list(rows).to_iterable_dataset()

        return _make

    @pytest.fixture
    def patch_load_dataset(self, mocker, make_iterable_dataset):
        def _patch(rows: list[dict[str, Any]]):
            def _factory(*args: Any, **kwargs: Any):
                return make_iterable_dataset(rows)

            mocker.patch(
                "am_diag.loaders.dataset.nejm_qa.load_dataset",
                side_effect=_factory,
            )

        return _patch
