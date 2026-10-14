"""Integration tests for PubHealthBench freeform dataset loader — loads real data.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import OpenEndedSample
from am_diag.loaders.dataset.pubhealthbench_freeform import (
    PubHealthBenchFreeformDataset,
)

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_openended_invariants,
    assert_split_value,
)


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestPubHealthBenchFreeformDatasetIntegration:
    """Integration tests — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        """Loader streams real PubHealthBench freeform samples."""
        samples = PubHealthBenchFreeformDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, OpenEndedSample) for s in samples)

    def test_invariants(self):
        """All samples have correct dataset, split, and non-empty question."""
        samples = PubHealthBenchFreeformDataset().load(limit=self.SAMPLE_LIMIT)
        assert_openended_invariants(samples)
        assert_dataset_name(samples, "pubhealthbench_freeform")
        assert_split_value(samples, "test")

    def test_reference_answer_populated(self):
        """reference_answer is a non-empty string for all loaded samples."""
        samples = PubHealthBenchFreeformDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert isinstance(s.reference_answer, str), (
                f"Sample {s.sample_id}: reference_answer not a string"
            )
            assert s.reference_answer, f"Sample {s.sample_id}: empty reference_answer"

    def test_metadata_fields_populated(self):
        """Metadata has question_id, category, source, retrieved_context_for_judge."""
        samples = PubHealthBenchFreeformDataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "question_id" in s.metadata, (
                f"Sample {s.sample_id}: missing question_id"
            )
            assert "category" in s.metadata, f"Sample {s.sample_id}: missing category"
            assert "source" in s.metadata, f"Sample {s.sample_id}: missing source"
            assert "retrieved_context_for_judge" in s.metadata, (
                f"Sample {s.sample_id}: missing retrieved_context_for_judge"
            )
            assert s.metadata["question_id"], f"Sample {s.sample_id}: empty question_id"

    def test_retrieved_context_in_metadata(self):
        """retrieved_context_for_judge is present in all samples."""
        samples = PubHealthBenchFreeformDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert "retrieved_context_for_judge" in s.metadata
            assert isinstance(s.metadata["retrieved_context_for_judge"], str)

    def test_default_split_is_test(self):
        """default_split is 'test'."""
        loader = PubHealthBenchFreeformDataset()
        assert loader.default_split == "test"

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(PubHealthBenchFreeformDataset)
