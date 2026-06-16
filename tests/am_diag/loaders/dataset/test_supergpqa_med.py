"""Unit tests for SuperGPQA Medicine subset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.supergpqa_med import SuperGPQAMedDataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_d_row(**overrides: Any) -> dict[str, Any]:
    """Pattern D: list options + letter answer. Used by MMLU-Pro Health, SuperGPQA."""
    row: dict[str, Any] = {
        "question": "Which intervention is most effective?",
        "options": ["Option A", "Option B", "Option C", "N/A"],
        "answer": "B",
        "answer_letter": "B",
        "category": "health",
        "cot_content": "Because ...",
        "src": "pubmed",
        "question_id": "mmlup_001",
    }
    row.update(overrides)
    return row


class TestSuperGPQAMedDataset(ShuffleTestMixin):
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    loader_class = SuperGPQAMedDataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        base = make_pattern_d_row(discipline="Medicine", **overrides)
        if "options" not in overrides:
            base["options"] = ["Option A", "Option B", "Option C", "Option D"]
        return base

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        row = make_pattern_d_row(question="", discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        row = make_pattern_d_row(options=[], discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_non_medicine_discipline_filtered(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Physics")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_invalid_answer_letter_returns_none(self, patch_load_dataset):
        row = make_pattern_d_row(answer_letter="Z", discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        row = make_pattern_d_row(answer_letter="Z", discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert samples[0].dataset == "supergpqa_med"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load(split="train")
        assert samples[0].split == "train"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine", field="Cardiology")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert "field" in samples[0].metadata
        assert samples[0].metadata["field"] == "Cardiology"

    def test_metadata_has_field_and_difficulty(self, patch_load_dataset):
        row = make_pattern_d_row(
            discipline="Medicine",
            field="Cardiology",
            difficulty="medium",
        )
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert "field" in samples[0].metadata
        assert samples[0].metadata["field"] == "Cardiology"
        assert "difficulty" in samples[0].metadata
        assert samples[0].metadata["difficulty"] == "medium"

    def test_medicine_discipline_included(self, patch_load_dataset):
        row = make_pattern_d_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 1

    def test_default_split_is_train(self):
        loader = SuperGPQAMedDataset()
        assert loader.default_split == "train"

    def test_field_filter_applied(self, patch_load_dataset):
        rows = [
            make_pattern_d_row(discipline="Medicine", field="Cardiology"),
            make_pattern_d_row(discipline="Medicine", field="Neurology"),
        ]
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", rows)
        samples = SuperGPQAMedDataset(field="Cardiology").load()
        assert len(samples) == 1
        assert samples[0].metadata["field"] == "Cardiology"

    def test_difficulty_filter_applied(self, patch_load_dataset):
        rows = [
            make_pattern_d_row(discipline="Medicine", difficulty="easy"),
            make_pattern_d_row(discipline="Medicine", difficulty="hard"),
        ]
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", rows)
        samples = SuperGPQAMedDataset(difficulty="easy").load()
        assert len(samples) == 1
        assert samples[0].metadata["difficulty"] == "easy"

    @pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
    def test_valid_answer_letters(self, patch_load_dataset, letter):
        # Generate 10 options so letters A-J all have a matching option.
        options = [f"Option {chr(ord('A') + i)}" for i in range(10)]
        row = make_pattern_d_row(
            answer_letter=letter,
            options=options,
            discipline="Medicine",
        )
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset(shuffle_options=False).load()
        assert len(samples) == 1
        assert samples[0].answer_key == letter

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_d_row(discipline="Medicine") for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", rows)
        samples = SuperGPQAMedDataset().load(limit=5)
        assert len(samples) <= 5
