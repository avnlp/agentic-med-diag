"""Integration tests for SuperGPQA Medicine subset loader — loads real data.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.supergpqa_med import SuperGPQAMedDataset

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


class TestSuperGPQAMedDatasetIntegration:
    """Integration tests for SuperGPQAMedDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        samples = SuperGPQAMedDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "supergpqa_med")

    def test_split_field(self):
        samples = SuperGPQAMedDataset().load(split="train", limit=5)
        assert_split_value(samples, "train")

    def test_split_field_default(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "train")

    def test_discipline_is_medicine(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.metadata.get("discipline") == "Medicine", (
                f"Sample {s.sample_id}: discipline={s.metadata.get('discipline')!r}"
            )

    def test_has_at_least_4_options(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_options_not_empty(s, min_options=4)

    def test_question_format(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_shuffle_preserves_originals(self):
        assert_shuffle_works(SuperGPQAMedDataset)

    def test_no_shuffle(self):
        loader = SuperGPQAMedDataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_metadata_fields(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "field" in s.metadata
            assert "difficulty" in s.metadata
            assert "discipline" in s.metadata
            assert "source" in s.metadata

    def test_question_stem_and_options_not_empty(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_limit_zero_returns_empty(self):
        assert_limit_zero_returns_empty(SuperGPQAMedDataset)

    def test_sample_ids_are_sequential(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for i, s in enumerate(samples):
            assert s.sample_id == str(i)

    def test_answer_key_in_options(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.answer_key in s.options, (
                f"Sample {s.sample_id}: answer_key {s.answer_key!r} not in options keys {set(s.options.keys())}"
            )

    def test_difficulty_in_metadata(self):
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.metadata["difficulty"] in ("easy", "middle", "hard")
