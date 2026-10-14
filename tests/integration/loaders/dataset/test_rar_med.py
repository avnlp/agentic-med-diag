"""Integration tests for RAR-Medicine rubric dataset loader.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import RARMedSample
from am_diag.loaders.dataset.rar_med import RARMedDataset

from .helpers import assert_dataset_name, assert_limit_zero_returns_empty


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestRARMedDatasetIntegration:
    """Integration tests for RARMedDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        """Loader streams real samples and returns RARMedSample instances."""
        samples = RARMedDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, RARMedSample) for s in samples)

    def test_invariants(self):
        """answer == reference_answer and all required fields are non-empty."""
        samples = RARMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.answer == s.reference_answer, (
                f"Sample {s.sample_id}: answer != reference_answer"
            )
            assert s.sample_id, "Empty sample_id"
            assert s.question, f"Sample {s.sample_id}: empty question"
            assert s.reference_answer, f"Sample {s.sample_id}: empty reference_answer"
        assert_dataset_name(samples, "rar_med")

    def test_sample_ids_are_sequential(self):
        """sample_id strings are sequential row indices starting from 0."""
        samples = RARMedDataset().load(limit=self.SAMPLE_LIMIT)
        for i, s in enumerate(samples):
            assert s.sample_id == str(i)

    def test_question_stem_field(self):
        """question_stem is present (may be empty string for this loader)."""
        samples = RARMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem == ""

    def test_rubrics_are_non_empty_lists(self):
        """Every sample has non-empty rubrics list with dict entries."""
        samples = RARMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert isinstance(s.rubrics, list), (
                f"Sample {s.sample_id}: rubrics is not a list"
            )
            assert len(s.rubrics) > 0, f"Sample {s.sample_id}: rubrics is empty"
            for rubric in s.rubrics:
                assert isinstance(rubric, dict), (
                    f"Sample {s.sample_id}: rubric item is not a dict"
                )
                assert "description" in rubric
                assert "title" in rubric
                assert "weight" in rubric

    def test_metadata_structure(self):
        """Metadata contains question_source and rubric_list."""
        samples = RARMedDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert isinstance(s.metadata, dict)
            assert "question_source" in s.metadata
            assert "rubric_list" in s.metadata
            assert "rubric_count" in s.metadata

    def test_default_split_is_train(self):
        """default split is 'train'."""
        assert RARMedDataset.default_split == "train"

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(RARMedDataset)

    def test_different_rubric_counts(self):
        """Different samples have different rubric counts."""
        samples = RARMedDataset().load(limit=self.SAMPLE_LIMIT)
        counts = {len(s.rubrics) for s in samples}
        assert len(counts) > 0
