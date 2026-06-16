"""Integration tests for PubHealthBench MCQ dataset loader — loads real data.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.pubhealthbench import PubHealthBenchDataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_mcq_invariants,
    assert_mcq_options_not_empty,
    assert_shuffle_works,
    assert_split_value,
)


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestPubHealthBenchDatasetIntegration:
    """Integration tests for PubHealthBenchDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_reviewed_samples(self):
        """Load real PubHealthBench samples, return MCQSample instances."""
        samples = PubHealthBenchDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """answer == answer_key and answer_text matches options[answer_key]."""
        samples = PubHealthBenchDataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "pubhealthbench")

    def test_options_have_valid_keys(self):
        """Option keys are uppercase letters and answer_key is among them."""
        samples = PubHealthBenchDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert all(k in "ABCDEFGHIJ" for k in s.options), (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in s.options, (
                f"Sample {s.sample_id}: answer_key {s.answer_key!r} not in options"
            )

    def test_split_field(self):
        """split field reflects the requested split."""
        samples = PubHealthBenchDataset().load(split="test", limit=5)
        assert_split_value(samples, "test")

    def test_default_split_is_reviewed(self):
        """default split is 'reviewed'."""
        samples = PubHealthBenchDataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "reviewed")

    def test_metadata_fields_populated(self):
        """Metadata has question_id, category, and source."""
        samples = PubHealthBenchDataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "question_id" in s.metadata, (
                f"Sample {s.sample_id}: missing question_id"
            )
            assert "category" in s.metadata, f"Sample {s.sample_id}: missing category"
            assert "source" in s.metadata, f"Sample {s.sample_id}: missing source"
            assert s.metadata["question_id"], f"Sample {s.sample_id}: empty question_id"

    def test_shuffle_enabled(self):
        """When shuffle is on, options_original is populated."""
        assert_shuffle_works(PubHealthBenchDataset)

    def test_shuffle_disabled(self):
        """When shuffle is off, options_original is None."""
        loader = PubHealthBenchDataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None, (
                f"Sample {s.sample_id}: options_original should be None when shuffle is off"
            )

    def test_category_filter(self):
        """Filtering by category returns only samples with that category."""
        samples = PubHealthBenchDataset(category="radiation").load(limit=5)
        for s in samples:
            assert s.metadata["category"] == "radiation", (
                f"Sample {s.sample_id}: expected 'radiation', got {s.metadata['category']!r}"
            )

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(PubHealthBenchDataset)

    def test_question_stem_and_options_not_empty(self):
        """Every sample has a non-empty question stem and non-empty options."""
        samples = PubHealthBenchDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )
