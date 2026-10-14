"""Integration tests for MedMCQA dataset loader — loads real data via streaming.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

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


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestMedMCQADatasetIntegration:
    """Integration tests for MedMCQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        samples = MedMCQADataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "medmcqa")

    def test_options_always_4_keys(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C", "D"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C", "D")

    def test_split_field(self):
        samples = MedMCQADataset().load(split="train", limit=5)
        assert_split_value(samples, "train")

    def test_split_field_default_is_validation(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "validation")

    def test_metadata_fields(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "subject_name" in s.metadata
            assert "topic_name" in s.metadata
            assert "choice_type" in s.metadata
            assert "id" in s.metadata
            assert "exp" in s.metadata

    def test_question_format(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_shuffle_preserves_originals(self):
        assert_shuffle_works(MedMCQADataset)

    def test_no_shuffle(self):
        loader = MedMCQADataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_question_stem_and_options_not_empty(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_train_split_loads(self):
        samples = MedMCQADataset().load(split="train", limit=5)
        assert len(samples) == 5
        for s in samples:
            assert s.split == "train"
            assert s.dataset == "medmcqa"

    def test_limit_zero_returns_empty(self):
        assert_limit_zero_returns_empty(MedMCQADataset)

    def test_sample_ids_are_hf_ids(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.sample_id, "Empty sample_id"
            assert isinstance(s.sample_id, str)

    def test_choice_type_in_metadata(self):
        samples = MedMCQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.metadata["choice_type"] in ("single", "multi")
