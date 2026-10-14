"""Integration tests for MedMCQA dataset loader.

Covers MCQ invariants, option keys, split values, metadata keys, shuffling,
and train-split loading on live data.

These tests load live data from Hugging Face and need network access.
"""

from __future__ import annotations

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.medmcqa import MedMCQADataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_mcq_invariants,
    assert_mcq_options_not_empty,
    assert_mcq_question_format,
    assert_shuffle_works,
    assert_split_value,
)


class TestMedMCQADatasetIntegration:
    """Integration tests for MedMCQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_samples(self):
        """Test loading samples when a limit is specified."""
        samples = MedMCQADataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """Test the answer invariants and dataset name match sampled data."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "medmcqa")

    def test_options_always_4_keys(self):
        """Test that every sample has exactly four option keys (A-D)."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C", "D"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C", "D")

    def test_split_field(self):
        """Test that the split field reflects the requested split."""
        samples = MedMCQADataset().load(split="train", limit=5)
        assert_split_value(samples, "train")

    def test_split_field_default_is_validation(self):
        """Test that the default split is 'validation' when no split specified."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "validation")

    def test_metadata_fields(self):
        """Test metadata contains subject_name, topic_name, choice_type, and id."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "subject_name" in s.metadata
            assert "topic_name" in s.metadata
            assert "choice_type" in s.metadata
            assert "id" in s.metadata
            assert "exp" in s.metadata

    def test_question_format(self):
        """Test the question follows the expected Question/Choices/Answer format."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_shuffle_preserves_originals(self):
        """Test that shuffle preserves original option order in options_original."""
        assert_shuffle_works(MedMCQADataset)

    def test_no_shuffle(self):
        """Test that options_original is None when shuffle_options=False."""
        loader = MedMCQADataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_question_stem_and_options_not_empty(self):
        """Test that question_stem and all option texts are non-empty."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_train_split_loads(self):
        """Test that the train split returns samples with split='train'."""
        samples = MedMCQADataset().load(split="train", limit=5)
        assert len(samples) == 5
        for s in samples:
            assert s.split == "train"
            assert s.dataset == "medmcqa"

    def test_limit_zero_returns_empty(self):
        """Test that limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(MedMCQADataset)

    def test_sample_ids_are_hf_ids(self):
        """Test that sample_id is a non-empty string from HuggingFace."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.sample_id, "Empty sample_id"
            assert isinstance(s.sample_id, str)

    def test_choice_type_in_metadata(self):
        """Test that choice_type metadata is 'single' or 'multi'."""
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.metadata["choice_type"] in ("single", "multi")
