"""Integration tests for CareQA MCQ dataset loader — loads real data via streaming.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.careqa import CareQADataset

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


class TestCareQADatasetIntegration:
    """Integration tests for CareQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        """Loader streams real CareQA samples and returns MCQSample instances."""
        samples = CareQADataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """answer == answer_key and answer_text matches options[answer_key]."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "careqa")

    def test_options_always_has_4_keys(self):
        """CareQA always has exactly 4 options labeled A-D."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C", "D"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C", "D")

    def test_split_field(self):
        """split field reflects the requested split."""
        samples = CareQADataset().load(limit=5)
        assert_split_value(samples, "test")

    def test_question_format(self):
        """question is a formatted prompt with Question/Choices/Answer structure."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_metadata_has_subject_and_id(self):
        """metadata contains 'subject' and 'id' keys."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert "subject" in s.metadata, f"Sample {s.sample_id}: missing subject"
            assert "id" in s.metadata, f"Sample {s.sample_id}: missing id"

    def test_non_empty_fields(self):
        """Every sample has non-empty question, question_stem, option texts, answer."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question, f"Sample {s.sample_id}: empty question"
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )
            assert s.answer_text, f"Sample {s.sample_id}: empty answer_text"

    def test_shuffle_preserves_originals(self):
        """When shuffle is on, options_original is populated."""
        assert_shuffle_works(CareQADataset)

    def test_no_shuffle(self):
        """When shuffle is off, options_original is None."""
        loader = CareQADataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_sample_ids_are_strings(self):
        """sample_id values are strings."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert isinstance(s.sample_id, str), (
                f"Sample {s.sample_id}: sample_id is not str"
            )

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(CareQADataset)
