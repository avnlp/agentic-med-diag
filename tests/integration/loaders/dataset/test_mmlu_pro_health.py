"""Integration tests for MMLU-Pro Health dataset loader — loads real data via streaming.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.mmlu_pro_health import MMLUProHealthDataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_mcq_invariants,
    assert_mcq_options_not_empty,
    assert_shuffle_works,
    assert_split_value,
)


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestMMLUProHealthDatasetIntegration:
    """Integration tests for MMLUProHealthDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_health_samples(self):
        """Loader returns MCQSample instances with health category."""
        samples = MMLUProHealthDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)
        assert all(s.metadata.get("category", "").lower() == "health" for s in samples)

    def test_invariants(self):
        """answer == answer_key and answer_text matches options[answer_key]."""
        samples = MMLUProHealthDataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "mmlu_pro_health")

    def test_no_na_in_options(self):
        """N/A option values are filtered out."""
        samples = MMLUProHealthDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            for val in s.options.values():
                assert str(val).strip().upper() != "N/A", (
                    f"Sample {s.sample_id}: option contains N/A"
                )

    def test_options_have_valid_keys(self):
        """Option keys are uppercase letters and answer_key is valid."""
        samples = MMLUProHealthDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert all(k in "ABCDEFGHIJ" for k in s.options), (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in s.options, (
                f"Sample {s.sample_id}: answer_key {s.answer_key!r} not in options"
            )

    def test_split_field(self):
        """split field reflects the requested split."""
        samples = MMLUProHealthDataset().load(split="test", limit=5)
        assert_split_value(samples, "test")

    def test_default_split_is_test(self):
        """default split is 'test'."""
        samples = MMLUProHealthDataset().load(limit=5)
        assert_split_value(samples, "test")

    def test_metadata_fields_populated(self):
        """Metadata has category, cot_content, src, question_id."""
        samples = MMLUProHealthDataset().load(limit=self.SAMPLE_LIMIT)
        assert len(samples) > 0
        for s in samples:
            assert "category" in s.metadata, f"Sample {s.sample_id}: missing category"
            assert "cot_content" in s.metadata, (
                f"Sample {s.sample_id}: missing cot_content"
            )
            assert "src" in s.metadata, f"Sample {s.sample_id}: missing src"
            assert "question_id" in s.metadata, (
                f"Sample {s.sample_id}: missing question_id"
            )

    def test_category_is_health(self):
        """All loaded samples have category='health'."""
        samples = MMLUProHealthDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.metadata.get("category", "").lower() == "health", (
                f"Sample {s.sample_id}: category={s.metadata.get('category', '')!r}"
            )

    def test_shuffle_enabled(self):
        """When shuffle is on, options_original is populated."""
        assert_shuffle_works(MMLUProHealthDataset)

    def test_shuffle_disabled(self):
        """When shuffle is off, options_original is None."""
        loader = MMLUProHealthDataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None, (
                f"Sample {s.sample_id}: options_original should be None"
            )

    def test_few_shot_populated(self):
        """Few-shot examples are populated after load with correct split."""
        loader = MMLUProHealthDataset(num_few_shot=3)
        loader.load(limit=10)
        assert 1 <= len(loader.few_shot_examples) <= 3
        for fs in loader.few_shot_examples:
            assert fs.dataset == "mmlu_pro_health"
            assert fs.split == "validation"
            assert fs.metadata.get("category", "").lower() == "health"

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(MMLUProHealthDataset)

    def test_question_stem_and_options_not_empty(self):
        """Every sample has a non-empty question stem and non-empty options."""
        samples = MMLUProHealthDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )
