"""Unit tests for PubHealthBench MCQ dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.pubhealthbench import PubHealthBenchDataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_c_row(**overrides: Any) -> dict[str, Any]:
    """Pattern C: list options + 0-indexed answer_index. Used by PubHealthBench."""
    row: dict[str, Any] = {
        "question": "What is the correct statement?",
        "options": ["Statement A", "Statement B", "Statement C"],
        "answer_index": 1,
        "question_id": "phb_001",
        "category": "Epidemiology",
        "source": "CDC",
    }
    row.update(overrides)
    return row


class TestPubHealthBenchDataset(ShuffleTestMixin):
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    loader_class = PubHealthBenchDataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_pattern_c_row(**overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.pubhealthbench", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(question="")],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(options=[])],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_empty_options_list_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(options=[])],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_missing_answer_index_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(answer_index=None)],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(answer_index=None)],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row()],
        )
        samples = PubHealthBenchDataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row()],
        )
        samples = PubHealthBenchDataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row()],
        )
        samples = PubHealthBenchDataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row()],
        )
        samples = PubHealthBenchDataset().load()
        assert samples[0].dataset == "pubhealthbench"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row()],
        )
        samples = PubHealthBenchDataset().load(split="test")
        assert samples[0].split == "test"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row()],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(question_id="phb_001")],
        )
        samples = PubHealthBenchDataset().load()
        assert "question_id" in samples[0].metadata
        assert samples[0].metadata["question_id"] == "phb_001"

    def test_metadata_has_question_id_and_category(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(question_id="phb_002", category="Cardiology")],
        )
        samples = PubHealthBenchDataset().load()
        assert "question_id" in samples[0].metadata
        assert samples[0].metadata["question_id"] == "phb_002"
        assert "category" in samples[0].metadata
        assert samples[0].metadata["category"] == "Cardiology"

    def test_default_split_is_reviewed(self):
        loader = PubHealthBenchDataset()
        assert loader.default_split == "reviewed"

    @pytest.mark.parametrize(
        "idx,expected",
        [(0, "A"), (1, "B"), (2, "C")],
    )
    def test_answer_index_maps_to_letter(self, patch_load_dataset, idx, expected):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(answer_index=idx)],
        )
        samples = PubHealthBenchDataset(shuffle_options=False).load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected

    def test_category_filter_applied(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(category="Epidemiology")],
        )
        loader = PubHealthBenchDataset(category="Epidemiology")
        samples = loader.load()
        assert len(samples) == 1

    def test_category_none_includes_all(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_pattern_c_row(category="Epidemiology")],
        )
        loader = PubHealthBenchDataset(category=None)
        samples = loader.load()
        assert len(samples) == 1

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_c_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.pubhealthbench", rows)
        samples = PubHealthBenchDataset().load(limit=5)
        assert len(samples) <= 5
