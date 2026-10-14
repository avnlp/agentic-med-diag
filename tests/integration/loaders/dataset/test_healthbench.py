"""Integration tests for HealthBench rubric-scored dataset loader — loads real data.

These tests require network access to HuggingFace Datasets.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import (
    ConsensusCriterion,
    ConversationTurn,
    RubricCriterion,
    RubricSample,
)
from am_diag.loaders.dataset.healthbench import HealthBenchDataset

from .helpers import assert_dataset_name, assert_limit_zero_returns_empty


pytestmark = [pytest.mark.integration, pytest.mark.enable_socket]


class TestHealthBenchDatasetIntegration:
    """Integration tests for HealthBenchDataset — requires network access."""

    SAMPLE_LIMIT = 5

    def test_load_5_all_variant(self):
        """Load real HealthBench samples, return RubricSample instances."""
        samples = HealthBenchDataset(variant="all").load(limit=5)
        assert 1 <= len(samples) <= 5
        assert all(isinstance(s, RubricSample) for s in samples)

    def test_invariants(self):
        """answer is always empty string for HealthBench."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.answer == "", f"Sample {s.sample_id}: answer={s.answer!r}"
        assert_dataset_name(samples, "healthbench")

    def test_sample_types(self):
        """Check nested types: ConversationTurn, RubricCriterion, ConsensusCriterion."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            for turn in s.conversation:
                assert isinstance(turn, ConversationTurn)
            for rubric in s.rubrics:
                assert isinstance(rubric, RubricCriterion)
            for cc in s.consensus_criteria:
                if cc is not None:
                    assert isinstance(cc, ConsensusCriterion)

    def test_list_lengths_match(self):
        """All parallel lists (criteria, points_list, axes, etc.) same length."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            n = len(s.criteria)
            assert n > 0, f"Sample {s.sample_id}: empty criteria"
            assert len(s.criterion_ids) == n
            assert len(s.points_list) == n
            assert len(s.axes) == n
            assert len(s.consensus_criteria) == n

    def test_criterion_ids_are_16_hex_chars(self):
        """criterion_ids are 16-character hex strings (blake2b 8-byte digest)."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            for cid in s.criterion_ids:
                assert len(cid) == 16, (
                    f"Sample {s.sample_id}: cid={cid!r} len={len(cid)}"
                )
                int(cid, 16)  # assert valid hex

    def test_conversation_non_empty(self):
        """Every sample has at least one conversation turn."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert len(s.conversation) >= 1, f"Sample {s.sample_id}: no turns"
            for turn in s.conversation:
                assert turn.role, f"Sample {s.sample_id}: empty role"
                assert turn.content, f"Sample {s.sample_id}: empty content"

    def test_rubrics_non_empty(self):
        """Every sample has at least one rubric criterion."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert len(s.rubrics) >= 1, f"Sample {s.sample_id}: no rubrics"
            for r in s.rubrics:
                assert r.criterion, f"Sample {s.sample_id}: empty criterion text"

    def test_hard_variant_loads(self):
        """'hard' variant loads and sets variant='hard'."""
        samples = HealthBenchDataset(variant="hard").load(limit=3)
        assert all(s.variant == "hard" for s in samples)

    def test_consensus_variant_loads(self):
        """'consensus' variant loads and sets variant='consensus'."""
        samples = HealthBenchDataset(variant="consensus").load(limit=3)
        assert all(s.variant == "consensus" for s in samples)

    def test_all_variant_loads(self):
        """'all' variant loads and sets variant='all'."""
        samples = HealthBenchDataset(variant="all").load(limit=3)
        assert all(s.variant == "all" for s in samples)

    def test_theme_is_present(self):
        """theme is a non-empty string for every sample."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert s.theme, f"Sample {s.sample_id}: missing theme"

    def test_sample_id_is_string(self):
        """sample_id is a non-empty string (UUID from prompt_id)."""
        samples = HealthBenchDataset(variant="all").load(limit=self.SAMPLE_LIMIT)
        for s in samples:
            assert isinstance(s.sample_id, str) and s.sample_id, (
                f"Expected non-empty string sample_id, got {s.sample_id!r}"
            )

    def test_limit_zero_returns_empty(self):
        """limit=0 returns an empty list."""
        assert_limit_zero_returns_empty(lambda: HealthBenchDataset(variant="all"))
