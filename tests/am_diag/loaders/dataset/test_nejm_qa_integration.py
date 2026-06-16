"""Integration tests for NEJM Q&A dataset loader.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.nejm_qa import NEJMQADataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_mcq_invariants,
    assert_mcq_options_not_empty,
    assert_mcq_question_format,
    assert_shuffle_works,
    assert_split_value,
)


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestNEJMQADatasetIntegration:
    """Integration tests for NEJMQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        """Loader streams real NEJM QA samples and returns MCQSample instances."""
        samples = NEJMQADataset().load(split="internal_medicine", limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """Core invariants: answer == answer_key, answer_text matches options."""
        samples = NEJMQADataset(shuffle_options=False).load(
            split="internal_medicine",
            limit=self.SAMPLE_LIMIT,
        )
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "nejm_qa")

    def test_multi_answer_invariants(self):
        """Multi-answer invariants: answer_keys, answer_texts, is_multi_answer."""
        samples = NEJMQADataset(shuffle_options=False).load(split="all", limit=200)
        multi = [s for s in samples if s.is_multi_answer]
        assert len(multi) > 0, "No multi-answer questions found"
        for s in multi:
            assert s.answer_key == s.answer_keys[0]
            assert s.answer_text == s.answer_texts[0]
            for ki, k in enumerate(s.answer_keys):
                assert k in s.options, (
                    f"Sample {s.sample_id}: answer_key[{ki}]={k!r} not in options"
                )
                assert s.answer_texts[ki] == s.options[k], (
                    f"Sample {s.sample_id}: answer_texts[{ki}] mismatch for key {k!r}"
                )

    def test_options_have_4_keys(self):
        """NEJM QA questions have 4 options labeled A-D."""
        samples = NEJMQADataset(shuffle_options=False).load(
            split="internal_medicine",
            limit=self.SAMPLE_LIMIT,
        )
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C", "D"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C", "D")

    def test_split_field(self):
        """split field reflects the requested specialty."""
        samples = NEJMQADataset().load(split="pediatrics", limit=5)
        assert_split_value(samples, "pediatrics")

    def test_split_field_default(self):
        """default split is 'all'."""
        assert NEJMQADataset.default_split == "all"

    def test_sample_ids_format(self):
        """sample_id includes specialty prefix and row index."""
        samples = NEJMQADataset().load(split="psychiatry", limit=5)
        for s in samples:
            assert s.sample_id.startswith("psychiatry_")
            assert s.sample_id.split("_")[-1].isdigit()

    def test_question_format(self):
        """question is a formatted prompt with Question/Choices/Answer structure."""
        samples = NEJMQADataset(shuffle_options=False).load(
            split="internal_medicine",
            limit=self.SAMPLE_LIMIT,
        )
        for s in samples:
            assert_mcq_question_format(s)

    def test_multi_answer_prompt_format(self):
        """Multi-answer questions use 'A or B,C' instruction."""
        samples = NEJMQADataset(shuffle_options=False).load(split="all", limit=200)
        multi = [s for s in samples if s.is_multi_answer]
        assert len(multi) > 0, "No multi-answer questions found"
        for s in multi:
            assert "letter(s) of the correct" in s.question
            assert "option(s)" in s.question

    def test_shuffle_preserves_originals(self):
        """When shuffle is on, options_original is populated."""
        assert_shuffle_works(NEJMQADataset)

    def test_no_shuffle(self):
        """When shuffle is off, options_original is None."""
        loader = NEJMQADataset(shuffle_options=False)
        samples = loader.load(split="internal_medicine", limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_load_all_specialties(self):
        """Loading split='all' loads samples from multiple specialties."""
        samples = NEJMQADataset(shuffle_options=False).load(split="all", limit=50)
        assert len(samples) > 0
        specialties = {s.split for s in samples}
        assert len(specialties) > 0

    def test_question_stem_and_options_not_empty(self):
        """Every sample has a non-empty question stem and non-empty options."""
        samples = NEJMQADataset(shuffle_options=False).load(
            split="internal_medicine",
            limit=self.SAMPLE_LIMIT,
        )
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_metadata_has_specialty(self):
        """Metadata contains specialty and answer_raw."""
        samples = NEJMQADataset(shuffle_options=False).load(
            split="general_surgery",
            limit=5,
        )
        for s in samples:
            assert s.metadata.get("specialty") == "general_surgery"
            assert "answer_raw" in s.metadata
            assert "answer_letters" in s.metadata

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(NEJMQADataset)

    def test_shuffle_changes_answer_key(self):
        """Shuffled answer_key differs from original for at least some samples."""
        no_shuffle = NEJMQADataset(shuffle_options=False).load(
            split="internal_medicine",
            limit=self.SAMPLE_LIMIT,
        )
        shuffle = NEJMQADataset(shuffle_options=True, shuffle_seed=42).load(
            split="internal_medicine",
            limit=self.SAMPLE_LIMIT,
        )
        ns_keys = [s.answer_key for s in no_shuffle]
        s_keys = [s.answer_key for s in shuffle]
        assert ns_keys != s_keys, "No answer keys changed after shuffle"
