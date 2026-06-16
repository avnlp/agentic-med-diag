"""Integration tests for MMLU-Med dataset loader — loads real data via streaming.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.mmlu_med import MMLUMedDataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_mcq_invariants,
    assert_mcq_options_not_empty,
    assert_shuffle_works,
    assert_split_value,
)


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestMMLUMedDatasetIntegration:
    """Integration tests for MMLUMedDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_all_subjects_returns_approx_1089(self):
        """All 6 subjects combined yield ~1089 total test samples."""
        samples = MMLUMedDataset().load()
        assert 1000 <= len(samples) <= 1200

    def test_subject_filter_returns_subset(self):
        """Filtering by a single subject returns fewer samples than all."""
        all_samples = MMLUMedDataset().load()
        anatomy_samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(anatomy_samples) < len(all_samples)
        assert all(s.metadata["subject"] == "anatomy" for s in anatomy_samples)

    def test_load_5_samples(self):
        """Loader returns MCQSample instances."""
        samples = MMLUMedDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """answer == answer_key and answer_text matches options[answer_key]."""
        samples = MMLUMedDataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "mmlu_med")

    def test_options_always_4_keys(self):
        """MMLU-Med always has exactly 4 options labeled A-D."""
        samples = MMLUMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C", "D"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C", "D")

    def test_split_field(self):
        """split field reflects the requested split."""
        samples = MMLUMedDataset().load(split="test", limit=5)
        assert_split_value(samples, "test")

    def test_default_split_is_test(self):
        """default split is 'test'."""
        samples = MMLUMedDataset().load(limit=5)
        assert_split_value(samples, "test")

    def test_subject_in_metadata(self):
        """Every sample has a subject in metadata."""
        samples = MMLUMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert "subject" in s.metadata, f"Sample {s.sample_id}: missing subject"
            assert s.metadata["subject"], f"Sample {s.sample_id}: empty subject"

    def test_sample_id_prefix(self):
        """sample_id starts with subject prefix (e.g. 'anatomy_0')."""
        samples = MMLUMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            subject = s.metadata.get("subject", "")
            assert s.sample_id.startswith(f"{subject}_"), (
                f"Sample {s.sample_id}: expected prefix '{subject}_'"
            )

    def test_shuffle_enabled(self):
        """When shuffle is on, options_original is populated."""
        assert_shuffle_works(MMLUMedDataset)

    def test_shuffle_disabled(self):
        """When shuffle is off, options_original is None."""
        loader = MMLUMedDataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None, (
                f"Sample {s.sample_id}: options_original should be None"
            )

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(MMLUMedDataset)

    def test_question_stem_and_options_not_empty(self):
        """Every sample has a non-empty question stem and non-empty options."""
        samples = MMLUMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_subject_filter_sample_ids(self):
        """Filtered samples have correct subject prefix in sample_id."""
        anatomy_samples = MMLUMedDataset(subjects=["anatomy"]).load(limit=5)
        for s in anatomy_samples:
            assert s.sample_id.startswith("anatomy_")
            assert s.metadata["subject"] == "anatomy"
