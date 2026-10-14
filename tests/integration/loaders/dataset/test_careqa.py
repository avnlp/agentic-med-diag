"""Integration tests for CareQA MCQ dataset loader.

Covers column-option format (op1/op2/op3/op4), cop-to-letter mapping,
subject-metadata preservation, and shuffle invariants.
"""

from __future__ import annotations

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


class TestCareQADatasetIntegration:
    """Integration tests for CareQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_samples(self):
        """Test loading samples when a limit is specified."""
        samples = CareQADataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        """Test the answer == answer_key and answer_text matches options[answer_key]."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "careqa")

    def test_options_always_has_4_keys(self):
        """Test that CareQA always has exactly 4 options labeled A-D."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C", "D"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C", "D")

    def test_split_field(self):
        """Test that the split field reflects the requested split."""
        samples = CareQADataset().load(limit=5)
        assert_split_value(samples, "test")

    def test_question_format(self):
        """Test the question follows the Question/Choices/Answer format."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_metadata_has_subject_and_id(self):
        """Test that metadata contains 'subject' and 'id' keys."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert "subject" in s.metadata, f"Sample {s.sample_id}: missing subject"
            assert "id" in s.metadata, f"Sample {s.sample_id}: missing id"

    def test_non_empty_fields(self):
        """Test that every sample has non-empty question, stem, options, and answer."""
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
        """Test that when shuffle is on, options_original is populated."""
        assert_shuffle_works(CareQADataset)

    def test_no_shuffle(self):
        """Test that when shuffle is off, options_original is None."""
        loader = CareQADataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_sample_ids_are_strings(self):
        """Test that sample_id values are strings."""
        samples = CareQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert isinstance(s.sample_id, str), (
                f"Sample {s.sample_id}: sample_id is not str"
            )

    def test_limit_zero_returns_empty(self):
        """Test that limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(CareQADataset)
