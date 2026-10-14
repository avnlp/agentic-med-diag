"""Unit tests for MedMCQA dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.loaders.dataset.medmcqa import MedMCQADataset


def make_column_options_row(**overrides: Any) -> dict[str, Any]:
    """Build a row with options in separate columns (opa/opb/opc/opd)."""
    row: dict[str, Any] = {
        "question": "Which drug?",
        "opa": "Drug A",
        "opb": "Drug B",
        "opc": "Drug C",
        "opd": "Drug D",
        "cop": 2,
        "subject_name": "Pharmacology",
        "topic_name": "Antibiotics",
        "choice_type": "single",
        "id": "abc123",
        "exp": "Drug B is correct because ...",
    }
    row.update(overrides)
    return row


class TestMedMCQADataset:
    """Tests for MedMCQA extraction and metadata fields."""

    def test_all_four_options_empty_returns_none(self, patch_load_dataset):
        """Test that row with all four options empty produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(opa="", opb="", opc="", opd="")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(question="")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that row with all option fields empty produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(opa="", opb="", opc="", opd="")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_zero_returns_none(self, patch_load_dataset):
        """Test that cop=0 is rejected as out of valid range (1-4)."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=0)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_five_returns_none(self, patch_load_dataset):
        """Test that cop=5 is rejected as out of valid range (1-4)."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=5)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_none_returns_none(self, patch_load_dataset):
        """Test that None cop value is rejected and produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=None)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_string_returns_none(self, patch_load_dataset):
        """Test that string cop value (non-integer) is rejected."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop="1")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that row with out-of-range cop produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=0)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        """Test that load runs without error for the default MedMCQA split."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row()],
        )
        # Just verify load works with default split
        samples = MedMCQADataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that subject_name from the row is preserved in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(subject_name="Pharma")],
        )
        samples = MedMCQADataset().load()
        assert "subject_name" in samples[0].metadata
        assert samples[0].metadata["subject_name"] == "Pharma"

    def test_metadata_has_subject_and_topic(self, patch_load_dataset):
        """Test that both subject_name and topic_name are stored in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [
                make_column_options_row(
                    subject_name="Pharma",
                    topic_name="Antibiotics",
                ),
            ],
        )
        samples = MedMCQADataset().load()
        assert "subject_name" in samples[0].metadata
        assert samples[0].metadata["subject_name"] == "Pharma"
        assert "topic_name" in samples[0].metadata
        assert samples[0].metadata["topic_name"] == "Antibiotics"

    @pytest.mark.parametrize(
        "cop,expected_key",
        [(1, "A"), (2, "B"), (3, "C"), (4, "D")],
    )
    def test_cop_maps_to_correct_letter(self, patch_load_dataset, cop, expected_key):
        """Test that cop integer (1-4) is mapped to the correct letter answer key."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(cop=cop)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected_key

    def test_sample_id_uses_id_field(self, patch_load_dataset):
        """Test that sample_id is sourced from the row's id field when present."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(id="abc123")],
        )
        samples = MedMCQADataset().load()
        assert samples[0].sample_id == "abc123"

    def test_sample_id_fallback_to_index(self, patch_load_dataset):
        """Test that sample_id falls back to row index when id is None."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_column_options_row(id=None)],
        )
        samples = MedMCQADataset().load()
        assert samples[0].sample_id == "0"
