"""Integration tests for SuperGPQA-Med dataset loader.

Covers the Medicine discipline filter, option and answer-key invariants,
metadata keys, and shuffling on live data.

These tests load live data from Hugging Face and need network access.
"""

from __future__ import annotations

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


class TestSuperGPQAMedDatasetIntegration:
    """Integration tests for SuperGPQAMedDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_samples(self):
        """Test loading samples when a limit is specified."""
        samples = SuperGPQAMedDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """Test the answer invariants and dataset name match sampled data."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "supergpqa_med")

    def test_split_field(self):
        """Test that the split field reflects the requested split."""
        samples = SuperGPQAMedDataset().load(split="train", limit=5)
        assert_split_value(samples, "train")

    def test_split_field_default(self):
        """Test that the default split is 'train' when no split specified."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "train")

    def test_discipline_is_medicine(self):
        """Test that all samples have discipline='Medicine'."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.metadata.get("discipline") == "Medicine", (
                f"Sample {s.sample_id}: discipline={s.metadata.get('discipline')!r}"
            )

    def test_has_at_least_4_options(self):
        """Test that each sample has at least 4 options."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_options_not_empty(s, min_options=4)

    def test_question_format(self):
        """Test the question follows the expected Question/Choices/Answer format."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_shuffle_preserves_originals(self):
        """Test that shuffle preserves original option order in options_original."""
        assert_shuffle_works(SuperGPQAMedDataset)

    def test_no_shuffle(self):
        """Test that options_original is None when shuffle_options=False."""
        loader = SuperGPQAMedDataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_metadata_fields(self):
        """Test that metadata contains field, difficulty, discipline, and source."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "field" in s.metadata
            assert "difficulty" in s.metadata
            assert "discipline" in s.metadata
            assert "source" in s.metadata

    def test_question_stem_and_options_not_empty(self):
        """Test that question_stem and all option texts are non-empty."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_limit_zero_returns_empty(self):
        """Test that limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(SuperGPQAMedDataset)

    def test_sample_ids_are_sequential(self):
        """Test that sample IDs are sequential integers starting from 0."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for i, s in enumerate(samples):
            assert s.sample_id == str(i)

    def test_answer_key_in_options(self):
        """Test that answer_key is always among available option keys."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.answer_key in s.options, (
                f"Sample {s.sample_id}: answer_key {s.answer_key!r} not in options keys {set(s.options.keys())}"
            )

    def test_difficulty_in_metadata(self):
        """Test that difficulty metadata is easy, middle, or hard."""
        samples = SuperGPQAMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.metadata["difficulty"] in ("easy", "middle", "hard")
