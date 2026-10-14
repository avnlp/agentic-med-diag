"""Integration tests for MedQA dataset loader.

Covers dict-format options, meta_info preservation (step1/step2&3),
mcq invariants, and multiple split support.
"""

from __future__ import annotations

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.medqa import MedQADataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_mcq_invariants,
    assert_mcq_options_not_empty,
    assert_mcq_question_format,
    assert_shuffle_works,
    assert_split_value,
)


class TestMedQADatasetIntegration:
    """Integration tests for MedQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_samples(self):
        """Test loading samples when a limit is specified."""
        samples = MedQADataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """Test the answer == answer_key and answer_text matches options[answer_key]."""
        samples = MedQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "medqa")

    def test_options_always_has_4_keys(self):
        """Test that MedQA always has exactly 4 options labeled A-D."""
        samples = MedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C", "D"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C", "D")

    def test_split_field(self):
        """Test that the split field reflects the requested split."""
        samples = MedQADataset().load(split="train", limit=5)
        assert_split_value(samples, "train")

    def test_split_field_default(self):
        """Test that the default split is 'test'."""
        samples = MedQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "test")

    def test_meta_info_in_metadata(self):
        """Test that meta_info is preserved in metadata dict."""
        samples = MedQADataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "meta_info" in s.metadata, f"Sample {s.sample_id}: missing meta_info"
            assert s.metadata["meta_info"] in ("step1", "step2&3"), (
                f"Sample {s.sample_id}: unexpected meta_info={s.metadata['meta_info']!r}"
            )

    def test_question_format(self):
        """Test the question follows the Question/Choices/Answer format."""
        samples = MedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_shuffle_preserves_originals(self):
        """Test that when shuffle is on, options_original is populated."""
        assert_shuffle_works(MedQADataset)

    def test_no_shuffle(self):
        """Test that when shuffle is off, options_original is None."""
        loader = MedQADataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_sample_ids_are_sequential(self):
        """Test that sample_id strings are sequential row indices starting from 0."""
        samples = MedQADataset().load(limit=self.SAMPLE_LIMIT)
        for i, s in enumerate(samples):
            assert s.sample_id == str(i)

    def test_question_stem_and_options_not_empty(self):
        """Test every sample has a non-empty question stem and options."""
        samples = MedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_train_split_loads(self):
        """Test that the train split can be loaded via streaming."""
        samples = MedQADataset().load(split="train", limit=5)
        assert len(samples) == 5
        for s in samples:
            assert s.split == "train"
            assert s.dataset == "medqa"

    def test_limit_zero_returns_empty(self):
        """Test that limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(MedQADataset)
