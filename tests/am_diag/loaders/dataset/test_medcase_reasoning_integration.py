"""Integration tests for MedCaseReasoning open-ended dataset loader — loads real data.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import OpenEndedSample
from am_diag.loaders.dataset.medcase_reasoning import MedCaseReasoningDataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_openended_invariants,
    assert_split_value,
)


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestMedCaseReasoningDatasetIntegration:
    """Integration tests for MedCaseReasoningDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        """Loader streams real samples and returns OpenEndedSample instances."""
        samples = MedCaseReasoningDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, OpenEndedSample) for s in samples)

    def test_invariants(self):
        """answer == reference_answer and required fields are populated."""
        samples = MedCaseReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        assert_openended_invariants(samples)
        assert_dataset_name(samples, "medcase_reasoning")

    def test_split_field_val(self):
        """split='val' returns samples with split='val'."""
        samples = MedCaseReasoningDataset().load(split="val", limit=5)
        assert_split_value(samples, "val")

    def test_split_field_test(self):
        """split='test' returns samples with split='test'."""
        samples = MedCaseReasoningDataset().load(split="test", limit=5)
        assert_split_value(samples, "test")

    def test_split_field_train(self):
        """split='train' returns samples with split='train'."""
        samples = MedCaseReasoningDataset().load(split="train", limit=5)
        assert_split_value(samples, "train")

    def test_non_empty_fields(self):
        """Every sample has non-empty question, reference_answer, reasoning_chain."""
        samples = MedCaseReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question, f"Sample {s.sample_id}: empty question"
            assert s.reference_answer, f"Sample {s.sample_id}: empty reference_answer"
            assert s.reasoning_chain is not None, (
                f"Sample {s.sample_id}: reasoning_chain is None"
            )
            assert s.reasoning_chain, f"Sample {s.sample_id}: empty reasoning_chain"

    def test_metadata_has_all_fields(self):
        """metadata contains pmcid, title, journal, article_link, publication_date."""
        samples = MedCaseReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            meta = s.metadata
            assert "pmcid" in meta, f"Sample {s.sample_id}: missing pmcid"
            assert "title" in meta, f"Sample {s.sample_id}: missing title"
            assert "journal" in meta, f"Sample {s.sample_id}: missing journal"
            assert "article_link" in meta, f"Sample {s.sample_id}: missing article_link"
            assert "publication_date" in meta, (
                f"Sample {s.sample_id}: missing publication_date"
            )
            assert meta["pmcid"], f"Sample {s.sample_id}: empty pmcid"
            assert meta["title"], f"Sample {s.sample_id}: empty title"

    def test_sample_ids_are_pmcids(self):
        """sample_id strings are PubMed Central IDs."""
        samples = MedCaseReasoningDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.sample_id.startswith("PMC"), (
                f"Sample {s.sample_id}: does not start with PMC"
            )

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(MedCaseReasoningDataset)
