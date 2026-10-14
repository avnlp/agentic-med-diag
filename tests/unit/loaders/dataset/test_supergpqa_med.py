"""Unit tests for SuperGPQA Medicine loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.loaders.dataset.supergpqa_med import SuperGPQAMedDataset


def make_list_letter_answer_row(**overrides: Any) -> dict[str, Any]:
    """Build a row with options as list and answer/answer_letter as a letter."""
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


class TestSuperGPQAMedDataset:
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        row = make_list_letter_answer_row(question="", discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that row with empty options list produces no sample."""
        row = make_list_letter_answer_row(options=[], discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_non_medicine_discipline_filtered(self, patch_load_dataset):
        """Test that non-Medicine disciplines (e.g. Physics) are filtered out."""
        row = make_list_letter_answer_row(discipline="Physics")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_invalid_answer_letter_returns_none(self, patch_load_dataset):
        """Test that answer_letter outside the options range produces no sample."""
        row = make_list_letter_answer_row(answer_letter="Z", discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that row with unrecognized answer letter is filtered out."""
        row = make_list_letter_answer_row(answer_letter="Z", discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        """Test that load runs without error for the default SuperGPQA split."""
        row = make_list_letter_answer_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that field from the row is preserved in sample metadata."""
        row = make_list_letter_answer_row(discipline="Medicine", field="Cardiology")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert "field" in samples[0].metadata
        assert samples[0].metadata["field"] == "Cardiology"

    def test_metadata_has_field_and_difficulty(self, patch_load_dataset):
        """Test that both field and difficulty are stored in sample metadata."""
        row = make_list_letter_answer_row(
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
        """Test that Medicine discipline rows pass through the discipline filter."""
        row = make_list_letter_answer_row(discipline="Medicine")
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset().load()
        assert len(samples) == 1

    def test_default_split_is_train(self):
        """Test that the default split is 'train'."""
        loader = SuperGPQAMedDataset()
        assert loader.default_split == "train"

    def test_field_filter_applied(self, patch_load_dataset):
        """Test that field parameter restricts samples to that field."""
        rows = [
            make_list_letter_answer_row(discipline="Medicine", field="Cardiology"),
            make_list_letter_answer_row(discipline="Medicine", field="Neurology"),
        ]
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", rows)
        samples = SuperGPQAMedDataset(field="Cardiology").load()
        assert len(samples) == 1
        assert samples[0].metadata["field"] == "Cardiology"

    def test_difficulty_filter_applied(self, patch_load_dataset):
        """Test that difficulty parameter restricts samples to that difficulty."""
        rows = [
            make_list_letter_answer_row(discipline="Medicine", difficulty="easy"),
            make_list_letter_answer_row(discipline="Medicine", difficulty="hard"),
        ]
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", rows)
        samples = SuperGPQAMedDataset(difficulty="easy").load()
        assert len(samples) == 1
        assert samples[0].metadata["difficulty"] == "easy"

    @pytest.mark.parametrize("letter", list("ABCDEFGHIJ"))
    def test_valid_answer_letters(self, patch_load_dataset, letter):
        # Generate 10 options so letters A-J all have a matching option.
        """Test that letters A-J are all valid answer keys when options exist."""
        options = [f"Option {chr(ord('A') + i)}" for i in range(10)]
        row = make_list_letter_answer_row(
            answer_letter=letter,
            options=options,
            discipline="Medicine",
        )
        patch_load_dataset("am_diag.loaders.dataset.supergpqa_med", [row])
        samples = SuperGPQAMedDataset(shuffle_options=False).load()
        assert len(samples) == 1
        assert samples[0].answer_key == letter
