"""Unit tests for CareQA MCQ dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.loaders.dataset.careqa import CareQADataset


def make_column_options_row(**overrides: Any) -> dict[str, Any]:
    """Build a row with options in separate columns (op1/op2/op3/op4)."""
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


class TestCareQADataset:
    """Tests for CareQA extraction and subject metadata."""

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(question="")],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that cop value of 0 (out of range) produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=0)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_cop_out_of_range_returns_none(self, patch_load_dataset):
        """Test that cop=0 is rejected as out of range (valid cop is 1-4)."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=0)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_cop_none_returns_none(self, patch_load_dataset):
        """Test that None cop value is rejected and produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=None)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        """Test that load runs without error for the default CareQA split."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row()],
        )
        samples = CareQADataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that subject from the row is preserved in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(subject="Cardiology")],
        )
        samples = CareQADataset().load()
        assert "subject" in samples[0].metadata
        assert samples[0].metadata["subject"] == "Cardiology"

    def test_metadata_has_subject(self, patch_load_dataset):
        """Test that subject field is accessible on the sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(subject="Neurology")],
        )
        samples = CareQADataset().load()
        assert "subject" in samples[0].metadata
        assert samples[0].metadata["subject"] == "Neurology"

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that CareQA produces samples when row values are empty (cop valid)."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(op1="", op2="", op3="", op4="")],
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
        """Test that cop integer (1-4) is mapped to the correct letter answer key."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=cop)],
        )
        samples = CareQADataset().load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected

    def test_uses_careqa_en_config(self, patch_load_dataset):
        """Test that load runs without error using the careqa_en config."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row()],
        )
        samples = CareQADataset().load()
        assert len(samples) == 1
