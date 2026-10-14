"""Integration tests for CareQA open-ended reasoning dataset loader.

Covers open-ended sample invariants and split values on live data.

These tests load live data from Hugging Face and need network access.
"""

from __future__ import annotations

from am_diag.common.data_models import OpenEndedSample
from am_diag.loaders.dataset.careqa_reasoning import CareQAReasoningDataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_openended_invariants,
    assert_split_value,
)


class TestCareQAReasoningDatasetIntegration:
    """Integration tests for CareQAReasoningDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_samples(self):
        """Test loading samples when a limit is specified."""
        samples = CareQAReasoningDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, OpenEndedSample) for s in samples)

    def test_invariants(self):
        """Test the answer == reference_answer and required fields are populated."""
        samples = CareQAReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        assert_openended_invariants(samples)
        assert_dataset_name(samples, "careqa_reasoning")

    def test_split_field(self):
        """Test that the split field reflects the requested split."""
        samples = CareQAReasoningDataset().load(limit=5)
        assert_split_value(samples, "test")

    def test_reasoning_chain_is_none(self):
        """Test that the reasoning_chain is None for CareQA reasoning dataset."""
        samples = CareQAReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.reasoning_chain is None, (
                f"Sample {s.sample_id}: expected None reasoning_chain, got {s.reasoning_chain!r}"
            )

    def test_metadata_has_subject(self):
        """Test that the metadata contains 'subject' key."""
        samples = CareQAReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert "subject" in s.metadata, f"Sample {s.sample_id}: missing subject"

    def test_sample_ids_are_sequential(self):
        """Test that sample_id strings are sequential row indices starting from 0."""
        samples = CareQAReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        for i, s in enumerate(samples):
            assert s.sample_id == str(i)

    def test_limit_zero_returns_empty(self):
        """Test that limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(CareQAReasoningDataset)
