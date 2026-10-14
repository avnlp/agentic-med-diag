"""Integration tests for MedXpertQA dataset loader — loads real data via streaming.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.medxpertqa import MedXpertQADataset

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


class TestMedXpertQADatasetIntegration:
    """Integration tests for MedXpertQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        samples = MedXpertQADataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "medxpertqa")

    def test_split_field(self):
        samples = MedXpertQADataset().load(split="test", limit=5)
        assert_split_value(samples, "test")

    def test_split_field_default(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "test")

    def test_metadata_fields(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "medical_task" in s.metadata
            assert "body_system" in s.metadata
            assert "question_type" in s.metadata
            assert "id" in s.metadata

    def test_question_format(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s)

    def test_answer_choices_stripped(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert "Answer Choices:" not in s.question, (
                f"Sample {s.sample_id}: question still contains 'Answer Choices:'"
            )
            assert "Answer Choices:" not in s.question_stem, (
                f"Sample {s.sample_id}: question_stem still contains 'Answer Choices:'"
            )

    def test_has_at_least_4_options(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_options_not_empty(s, min_options=4)

    def test_shuffle_preserves_originals(self):
        assert_shuffle_works(MedXpertQADataset)

    def test_no_shuffle(self):
        loader = MedXpertQADataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_question_stem_and_options_not_empty(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_limit_zero_returns_empty(self):
        assert_limit_zero_returns_empty(MedXpertQADataset)

    def test_question_type_filter_reasoning(self):
        samples = MedXpertQADataset(question_type="reasoning").load(
            limit=self.SAMPLE_LIMIT,
        )
        for s in samples:
            assert s.metadata["question_type"].lower() == "reasoning"

    def test_question_type_filter_understanding(self):
        samples = MedXpertQADataset(question_type="understanding").load(
            limit=self.SAMPLE_LIMIT,
        )
        for s in samples:
            assert s.metadata["question_type"].lower() == "understanding"

    def test_sample_ids_are_sequential(self):
        samples = MedXpertQADataset().load(limit=self.SAMPLE_LIMIT)
        for i, s in enumerate(samples):
            assert s.sample_id == f"Text-{i}"
