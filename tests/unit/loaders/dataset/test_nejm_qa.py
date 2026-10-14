"""Unit tests for NEJM AI Q&A MCQ dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.loaders.dataset.nejm_qa import NEJMQADataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_mcq_row(**overrides: Any) -> dict[str, Any]:
    """Return a default NEJM QA row dict for testing, merged with overrides."""
    row: dict[str, Any] = {
        "question": "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid",
        "answer": "B",
    }
    row.update(overrides)
    return row


class TestNEJMQADataset(ShuffleTestMixin):
    """Tests for NEJM QA extraction and specialty-split loading."""

    loader_class = NEJMQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        """Return a valid row for the shared shuffle tests."""
        return make_mcq_row(**overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [self.make_row()])
        return self.loader_class(**loader_kwargs).load(split="pediatrics")

    def test_skips_row_with_empty_question(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_qa", [make_mcq_row(question="")]
        )
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_empty_answer(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that row with empty answer field produces no sample."""
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [make_mcq_row(answer="")])
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer_letter(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that answer letter outside A-D range produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_qa", [make_mcq_row(answer="Z")]
        )
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_skips_row_without_inline_options(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that question without inline options (A. / B.) produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_qa",
            [make_mcq_row(question="Just a plain question?")],
        )
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_parsed_stem_and_options(self, patch_load_dataset, patch_single_specialty):
        """Test that inline options parse into question_stem and options dict."""
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [make_mcq_row()])
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

    def test_sample_id_includes_specialty(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that sample_id is prefixed with the specialty name."""
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [make_mcq_row()])
        samples = NEJMQADataset().load()
        assert samples[0].sample_id == "pediatrics_0"

    def test_specialty_in_metadata(self, patch_load_dataset, patch_single_specialty):
        """Test that the specialty is recorded in sample metadata."""
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [make_mcq_row()])
        samples = NEJMQADataset().load()
        assert samples[0].metadata["specialty"] == "pediatrics"

    def test_answer_raw_in_metadata(self, patch_load_dataset, patch_single_specialty):
        """Test that raw answer string and parsed letters are stored in metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_qa", [make_mcq_row(answer="C")]
        )
        samples = NEJMQADataset().load()
        assert samples[0].metadata["answer_raw"] == "C"
        assert samples[0].metadata["answer_letters"] == ["C"]

    def test_multi_answer_populates_all_fields(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that multi-answer has correct answer_keys, texts, and compat."""
        row = make_mcq_row(
            question="Which are primary colors?\nA. Red\nB. Green\nC. Blue\nD. Yellow",
            answer="A,C",
        )
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [row])
        samples = NEJMQADataset().load()
        assert len(samples) == 1
        s = samples[0]
        assert s.is_multi_answer is True
        assert s.answer_keys == ["A", "C"]
        assert s.answer_texts == ["Red", "Blue"]
        # Backward compatibility
        assert s.answer_key == "A"
        assert s.answer == "A"
        assert s.answer_text == "Red"
        # Invariants hold
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
        """Test that multi-answer uses plural 'letter(s)' phrasing."""
        row = make_mcq_row(
            question="Which are primary colors?\nA. Red\nB. Green\nC. Blue\nD. Yellow",
            answer="A,C",
        )
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [row])
        samples = NEJMQADataset().load()
        assert "letter(s) of the correct option(s)" in samples[0].question

    def test_single_answer_prompt_uses_singular_instruction(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that single-answer samples use 'only the letter' phrasing."""
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [make_mcq_row()])
        samples = NEJMQADataset().load()
        assert "only the letter" in samples[0].question
        assert "letter(s)" not in samples[0].question

    def test_multi_answer_skips_when_no_valid_letter(
        self,
        patch_load_dataset,
        patch_single_specialty,
    ):
        """Test that multi-answer row with invalid letters (X,Y) produces no sample."""
        row = make_mcq_row(
            question="Which?\nA. One\nB. Two",
            answer="X,Y",
        )
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [row])
        samples = NEJMQADataset().load()
        assert len(samples) == 0

    def test_multi_answer_shuffle_tracks_all_keys(
        self,
        patch_load_dataset,
    ):
        """Test that shuffled multi-answer preserves all key mappings and invariants."""
        row = make_mcq_row(
            question="Which are primary colors?\nA. Red\nB. Green\nC. Blue\nD. Yellow",
            answer="A,C",
        )
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", [row])
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

    def test_load_specific_specialty(self, patch_load_dataset):
        """Test that split='pediatrics' loads only pediatrics samples."""
        rows = [make_mcq_row()]
        patch_load_dataset("am_diag.loaders.dataset.nejm_qa", rows)
        samples = NEJMQADataset().load(split="pediatrics")
        assert len(samples) == 1
        assert samples[0].split == "pediatrics"

    def test_unknown_split_raises(self):
        """Test that unknown split value raises ValueError."""
        with pytest.raises(ValueError, match="Unknown split"):
            NEJMQADataset().load(split="nonexistent")

    def test_extract_stem_and_options_parses_correctly(self):
        r"""Test that inline 'Question?\nA. Opt\nB. Opt' parses to stem and options."""
        ds = NEJMQADataset()
        text = "What is 2+2?\nA. 3\nB. 4\nC. 5"
        result = ds._extract_stem_and_options(text)
        assert result is not None
        stem, options = result
        assert stem == "What is 2+2?"
        assert options == {"A": "3", "B": "4", "C": "5"}

    def test_extract_stem_and_options_returns_none_for_plain_text(self):
        """Test that plain text without inline options returns None."""
        ds = NEJMQADataset()
        assert ds._extract_stem_and_options("Just a plain question.") is None

    def test_extract_stem_and_options_requires_min_two_options(self):
        """Test that text with only one option line returns None."""
        ds = NEJMQADataset()
        text = "Test?\nA. Only one"
        assert ds._extract_stem_and_options(text) is None

    @pytest.fixture(autouse=True)
    def patch_single_specialty(self, mocker):
        """Pin specialties to a single entry for deterministic tests."""
        mocker.patch("am_diag.loaders.dataset.nejm_qa._SPECIALTIES", ["pediatrics"])

    def test_ids_and_shuffle_stable_across_specialty_selection(
        self, patch_load_dataset, mocker
    ):
        """Test that a specialty's samples are the same under 'all' and alone."""
        mocker.patch(
            "am_diag.loaders.dataset.nejm_qa._SPECIALTIES",
            ["pediatrics", "psychiatry"],
        )
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_qa",
            [make_mcq_row(), make_mcq_row()],
        )
        alone = NEJMQADataset().load(split="psychiatry")
        everything = NEJMQADataset().load(split="all")
        psychiatry = [s for s in everything if s.metadata["specialty"] == "psychiatry"]
        assert [s.sample_id for s in psychiatry] == [s.sample_id for s in alone]
        assert [s.options for s in psychiatry] == [s.options for s in alone]
