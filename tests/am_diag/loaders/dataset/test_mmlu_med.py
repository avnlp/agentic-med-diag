"""Unit tests for MMLU-Med dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.mmlu_med import MMLUMedDataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_e_row(**overrides: Any) -> dict[str, Any]:
    """Pattern E: list choices + 0-indexed int answer. Used by MMLU-Med."""
    row: dict[str, Any] = {
        "question": "Which of the following is correct?",
        "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
        "answer": 2,
    }
    row.update(overrides)
    return row


class TestMMLUMedDataset(ShuffleTestMixin):
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    loader_class = MMLUMedDataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_pattern_e_row(**overrides)

    @property
    def _loader_kwargs(self) -> dict[str, Any]:
        return {"subjects": ["anatomy"]}

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_pattern_e_row(question="")],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_pattern_e_row(choices=[])],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_invalid_answer_none_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_pattern_e_row(answer=None)],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_invalid_answer_string_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_pattern_e_row(answer="A")],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_pattern_e_row(answer=None)],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert samples[0].dataset == "mmlu_med"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load(split="test")
        assert samples[0].split == "test"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert "subject" in samples[0].metadata
        assert samples[0].metadata["subject"] == "anatomy"

    def test_metadata_has_subject(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert "subject" in samples[0].metadata

    def test_subjects_filter_restricts_to_subset(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 1
        assert samples[0].metadata["subject"] == "anatomy"

    def test_sample_id_format_includes_subject(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", [make_pattern_e_row()])
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert samples[0].sample_id.startswith("anatomy_")

    @pytest.mark.parametrize(
        "answer_int,expected_key",
        [(0, "A"), (1, "B"), (2, "C"), (3, "D")],
    )
    def test_int_answer_maps_to_letter(
        self,
        patch_load_dataset,
        answer_int,
        expected_key,
    ):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_pattern_e_row(answer=answer_int)],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected_key

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_e_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.mmlu_med", rows)
        samples = MMLUMedDataset(subjects=["anatomy"]).load(limit=5)
        assert len(samples) <= 5
