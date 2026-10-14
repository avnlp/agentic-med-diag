"""Integration tests for PubMedQA dataset loader — loads real data via streaming.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

from collections import Counter

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.pubmedqa import PubMedQADataset

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


class TestPubMedQADatasetIntegration:
    """Integration tests for PubMedQADataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_full_test_split_returns_500(self):
        samples = PubMedQADataset(shuffle_options=False).load()
        assert len(samples) == 500

    def test_load_small_sample(self):
        samples = PubMedQADataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_invariants(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_mcq_invariants(samples)
        assert_dataset_name(samples, "pubmedqa")

    def test_options_always_3_keys(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert set(s.options.keys()) == {"A", "B", "C"}, (
                f"Sample {s.sample_id}: unexpected keys {set(s.options.keys())}"
            )
            assert s.answer_key in ("A", "B", "C")

    def test_option_texts_are_yes_no_maybe(self):
        samples = PubMedQADataset(shuffle_options=False).load(limit=self.SAMPLE_LIMIT)
        expected = {"A": "Yes", "B": "No", "C": "Maybe"}
        for s in samples:
            assert s.options == expected, (
                f"Sample {s.sample_id}: unexpected options {s.options}"
            )

    def test_split_field(self):
        samples = PubMedQADataset().load(split="test", limit=5)
        assert_split_value(samples, "test")

    def test_split_field_default(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "test")

    def test_question_stem_contains_context(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem.startswith("Context:"), (
                f"Sample {s.sample_id}: question_stem does not start with 'Context:'"
            )
            assert "Question:" in s.question_stem

    def test_question_format(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert_mcq_question_format(s, has_context=True)

    def test_shuffle_preserves_originals(self):
        assert_shuffle_works(PubMedQADataset)

    def test_no_shuffle(self):
        loader = PubMedQADataset(shuffle_options=False)
        samples = loader.load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.options_original is None

    def test_metadata_fields(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert "pubid" in s.metadata
            assert "meshes" in s.metadata

    def test_sample_ids_are_pubids(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.sample_id, "Empty sample_id"
            assert s.sample_id.isdigit(), (
                f"Sample {s.sample_id}: expected numeric pubid"
            )

    def test_unique_sample_ids(self):
        samples = PubMedQADataset(shuffle_options=False).load()
        ids = [s.sample_id for s in samples]
        assert len(set(ids)) == 500

    def test_answer_distribution(self):
        samples = PubMedQADataset(shuffle_options=False).load()
        answers = Counter(s.answer_key for s in samples)
        assert answers["A"] + answers["B"] + answers["C"] == 500
        assert answers["A"] > answers["B"]
        assert answers["B"] > answers["C"]

    def test_question_stem_and_options_not_empty(self):
        samples = PubMedQADataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem, f"Sample {s.sample_id}: empty question_stem"
            assert_mcq_options_not_empty(s)
            assert all(v for v in s.options.values()), (
                f"Sample {s.sample_id}: empty option text"
            )

    def test_limit_zero_returns_empty(self):
        assert_limit_zero_returns_empty(PubMedQADataset)

    def test_train_split_loads(self):
        samples = PubMedQADataset().load(split="train", limit=5)
        assert len(samples) == 5
        for s in samples:
            assert s.split == "train"
            assert s.dataset == "pubmedqa"
