"""Unit tests for HealthBench rubric-scored dataset loader."""

from __future__ import annotations

import re
from typing import Any

import pytest

from am_diag.loaders.dataset.healthbench import HealthBenchDataset


def make_healthbench_row(**overrides: Any) -> dict[str, Any]:
    """Row factory for HealthBench rubric-scored samples."""
    row: dict[str, Any] = {
        "prompt_id": "hb_001",
        "prompt": [
            {"role": "user", "content": "I have a headache."},
            {"role": "assistant", "content": "Tell me more."},
        ],
        "rubrics": [
            {
                "criterion": "The response asks about duration.",
                "points": 1,
                "tags": ["level:example", "axis:accuracy", "cluster:"],
            },
            {
                "criterion": "Recommends emergency referral if needed.",
                "points": 2,
                "tags": [
                    "level:cluster",
                    "axis:safety",
                    "cluster:emergency_referrals_emergent_emergency_behavior",
                ],
            },
        ],
        "example_tags": ["theme:clinical_reasoning", "other:tag"],
    }
    row.update(overrides)
    return row


class TestHealthBenchDataset:
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    def test_unknown_variant_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            HealthBenchDataset(variant="invalid")

    def test_tag_parsing_extracts_axis(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        # First rubric has "axis:accuracy"
        assert samples[0].rubrics[0].axis == "accuracy"

    def test_tag_parsing_extracts_level(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        assert samples[0].rubrics[0].level == "example"

    def test_tag_parsing_extracts_cluster(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        assert (
            samples[0].rubrics[1].cluster
            == "emergency_referrals_emergent_emergency_behavior"
        )

    def test_criterion_ids_are_16_hex_chars(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        for cid in samples[0].criterion_ids:
            assert re.fullmatch(r"[0-9a-f]{16}", cid)

    def test_parallel_list_lengths_equal(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        s = samples[0]
        assert len(s.criteria) == len(s.points_list)
        assert len(s.criteria) == len(s.axes)
        assert len(s.criteria) == len(s.criterion_ids)
        assert len(s.criteria) == len(s.consensus_criteria)

    def test_consensus_criterion_looked_up_for_known_cluster(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        # Second rubric has cluster "emergency_referrals_emergent_emergency_behavior"
        cc = samples[0].consensus_criteria[1]
        assert cc is not None
        assert cc.theme  # non-empty
        assert cc.behavior_category
        assert cc.criterion

    def test_consensus_criterion_is_none_for_example_level(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        # First rubric has level="example" → consensus is None
        assert samples[0].consensus_criteria[0] is None

    def test_theme_extracted_from_example_tags(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        assert samples[0].theme == "clinical_reasoning"

    def test_answer_is_empty_string(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        assert samples[0].answer == ""

    def test_conversation_built_from_prompt(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        conv = samples[0].conversation
        assert len(conv) == 2
        assert conv[0].role == "user"
        assert conv[1].role == "assistant"

    def test_question_is_serialized_conversation(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant="all").load()
        assert "user: I have a headache." in samples[0].question
        assert "assistant: Tell me more." in samples[0].question

    @pytest.mark.parametrize(
        "variant,expected_repo,expected_split",
        [
            ("all", "neuralleap/healthbench-regular", "test"),
            ("hard", "neuralleap/healthbench-hard", "train"),
            ("consensus", "neuralleap/healthbench-consensus", "train"),
        ],
    )
    def test_variant_routes_to_correct_repo(
        self,
        variant,
        expected_repo,
        expected_split,
        patch_load_dataset,
    ):
        patch_load_dataset(
            "am_diag.loaders.dataset.healthbench",
            [make_healthbench_row()],
        )
        samples = HealthBenchDataset(variant=variant).load()
        assert len(samples) == 1

    def test_limit_param(self, patch_load_dataset):
        rows = [make_healthbench_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.healthbench", rows)
        samples = HealthBenchDataset(variant="all").load(limit=5)
        assert len(samples) <= 5
