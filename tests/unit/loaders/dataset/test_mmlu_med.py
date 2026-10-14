"""Unit tests for MMLU-Med dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.loaders.dataset.mmlu_med import MMLUMedDataset


def make_list_indexed_row(**overrides: Any) -> dict[str, Any]:
    """Build a row with choices as list and answer as 0-indexed int."""
    row: dict[str, Any] = {
        "question": "Which of the following is correct?",
        "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
        "answer": 2,
    }
    row.update(overrides)
    return row


class TestMMLUMedDataset:
    """Tests for MMLU-Med extraction and subject filtering."""

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_list_indexed_row(question="")],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that row with empty choices list produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_list_indexed_row(choices=[])],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_invalid_answer_none_returns_none(self, patch_load_dataset):
        """Test that None answer value is rejected and produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_list_indexed_row(answer=None)],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_invalid_answer_string_returns_none(self, patch_load_dataset):
        """Test that string answer (non-integer) is rejected for MMLU-Med format."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_list_indexed_row(answer="A")],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that row with None answer produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_list_indexed_row(answer=None)],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        """Test that load runs without error for the default MMLU-Med split."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med", [make_list_indexed_row()]
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that subject field is stored in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med", [make_list_indexed_row()]
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert "subject" in samples[0].metadata
        assert samples[0].metadata["subject"] == "anatomy"

    def test_metadata_has_subject(self, patch_load_dataset):
        """Test that subject field is present on the sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med", [make_list_indexed_row()]
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert "subject" in samples[0].metadata

    def test_subjects_filter_restricts_to_subset(self, patch_load_dataset):
        """Test that subjects=["anatomy"] limits loaded samples to that subject."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med", [make_list_indexed_row()]
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 1
        assert samples[0].metadata["subject"] == "anatomy"

    def test_sample_id_format_includes_subject(self, patch_load_dataset):
        """Test that sample_id is prefixed with the subject name."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med", [make_list_indexed_row()]
        )
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
        """Test that 0-indexed int answer maps to corresponding letter (0=A, 1=B)."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_med",
            [make_list_indexed_row(answer=answer_int)],
        )
        samples = MMLUMedDataset(subjects=["anatomy"]).load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected_key
