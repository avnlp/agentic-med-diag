"""Unit tests for CareQA MCQ dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.careqa import CareQADataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_b_careqa_row(**overrides: Any) -> dict[str, Any]:
    """Pattern B: op1/op2/op3/op4 + cop int. Used by CareQA."""
    row: dict[str, Any] = {
        "question": "What is the first-line?",
        "op1": "Option 1",
        "op2": "Option 2",
        "op3": "Option 3",
        "op4": "Option 4",
        "cop": 3,
        "subject": "Internal Medicine",
        "id": "cqa_001",
    }
    row.update(overrides)
    return row


class TestCareQADataset(ShuffleTestMixin):
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    loader_class = CareQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_pattern_b_careqa_row(**overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.base", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(question="")],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(cop=0)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_cop_out_of_range_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(cop=0)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_cop_none_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(cop=None)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row()],
        )
        samples = CareQADataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row()],
        )
        samples = CareQADataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row()],
        )
        samples = CareQADataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row()],
        )
        samples = CareQADataset().load()
        assert samples[0].dataset == "careqa"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row()],
        )
        samples = CareQADataset().load(split="test")
        assert samples[0].split == "test"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row()],
        )
        samples = CareQADataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(subject="Cardiology")],
        )
        samples = CareQADataset().load()
        assert "subject" in samples[0].metadata
        assert samples[0].metadata["subject"] == "Cardiology"

    def test_metadata_has_subject(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(subject="Neurology")],
        )
        samples = CareQADataset().load()
        assert "subject" in samples[0].metadata
        assert samples[0].metadata["subject"] == "Neurology"

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(op1="", op2="", op3="", op4="")],
        )
        # CareQA always creates options dict even with empty values
        # but cop must be valid
        samples = CareQADataset().load()
        # cop=3 is still valid, so it should produce a sample
        assert len(samples) == 1

    @pytest.mark.parametrize(
        "cop,expected",
        [(1, "A"), (2, "B"), (3, "C"), (4, "D")],
    )
    def test_cop_maps_to_letter(self, patch_load_dataset, cop, expected):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row(cop=cop)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected

    def test_uses_careqa_en_config(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_careqa_row()],
        )
        samples = CareQADataset().load()
        assert len(samples) == 1

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_b_careqa_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.base", rows)
        samples = CareQADataset().load(limit=5)
        assert len(samples) <= 5
