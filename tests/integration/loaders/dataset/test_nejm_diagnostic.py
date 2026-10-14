"""Integration tests for NEJM Diagnostic Reasoning dataset loader.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import OpenEndedSample
from am_diag.loaders.dataset.nejm_diagnostic import NEJMDiagnosticDataset

from .helpers import (
    assert_dataset_name,
    assert_limit_zero_returns_empty,
    assert_openended_invariants,
    assert_split_value,
)


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestNEJMDiagnosticDatasetIntegration:
    """Integration tests for NEJMDiagnosticDataset — requires network access."""

    SAMPLE_LIMIT = 10

    def test_load_5_samples(self):
        """Loader streams real samples and returns OpenEndedSample instances."""
        samples = NEJMDiagnosticDataset().load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, OpenEndedSample) for s in samples)

    def test_invariants(self):
        """answer == reference_answer and required fields are populated."""
        samples = NEJMDiagnosticDataset().load(limit=self.SAMPLE_LIMIT)
        assert_openended_invariants(samples)
        assert_dataset_name(samples, "nejm_diagnostic")

    def test_split_field(self):
        """split field reflects the requested split."""
        samples = NEJMDiagnosticDataset().load(split="nejm_test", limit=5)
        assert_split_value(samples, "nejm_test")

    def test_split_field_default(self):
        """default split is 'nejm_test'."""
        samples = NEJMDiagnosticDataset().load(limit=self.SAMPLE_LIMIT)
        assert_split_value(samples, "nejm_test")

    def test_sample_ids_are_sequential(self):
        """sample_id strings are sequential row indices starting from 0."""
        samples = NEJMDiagnosticDataset().load(limit=self.SAMPLE_LIMIT)
        for i, s in enumerate(samples):
            assert s.sample_id == str(i)

    def test_question_stem_field(self):
        """question_stem is present (may be empty string for this loader)."""
        samples = NEJMDiagnosticDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question_stem == ""

    def test_reasoning_chain_some_present(self):
        """Some samples have reasoning_chain, some may not."""
        samples = NEJMDiagnosticDataset().load(limit=self.SAMPLE_LIMIT)
        with_chain = sum(1 for s in samples if s.reasoning_chain)
        assert len(samples) > 0
        assert with_chain > 0, "No samples had reasoning_chain"

    def test_metadata_structure(self):
        """Metadata contains expected keys."""
        samples = NEJMDiagnosticDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert isinstance(s.metadata, dict)
            assert "specialty" in s.metadata
            assert "source" in s.metadata

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(NEJMDiagnosticDataset)

    def test_question_starts_with_case(self):
        """Questions typically start with 'Case' for NEJM CPC samples."""
        samples = NEJMDiagnosticDataset().load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.question, "Empty question"
