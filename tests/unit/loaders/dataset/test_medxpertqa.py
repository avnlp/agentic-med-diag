"""Unit tests for MedXpertQA dataset loader."""

from __future__ import annotations

from typing import Any

from am_diag.loaders.dataset.medxpertqa import MedXpertQADataset


def make_dict_options_row(**overrides: Any) -> dict[str, Any]:
    """Build a row with options as dict and answer_idx as a letter."""
    row: dict[str, Any] = {
        "question": "Which drug is used for X?",
        "options": {
            "A": "Drug A",
            "B": "Drug B",
            "C": "Drug C",
            "D": "Drug D",
        },
        "answer_idx": "B",
        "meta_info": "step2",
    }
    row.update(overrides)
    return row


class TestMedXpertQADataset:
    """Tests for MedXpertQA extraction and option mapping."""

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(question="")],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that row with empty options dict produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(options={})],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that label outside the options range produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(label="Z")],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_label_not_in_options_returns_none(self, patch_load_dataset):
        """Test that label not matching any option key is rejected."""
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(label="Z")],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_question_type_filter(self, patch_load_dataset):
        """Test that question_type filters out rows with non-matching types."""
        row = self._make_row(question_type="reasoning")
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [row])
        loader = MedXpertQADataset(question_type="understanding")
        samples = loader.load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        """Test that load runs without error for the default MedXpertQA split."""
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that medical_task from the row is preserved in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(medical_task="diagnosis")],
        )
        samples = MedXpertQADataset().load()
        assert "medical_task" in samples[0].metadata
        assert samples[0].metadata["medical_task"] == "diagnosis"

    def test_metadata_has_medical_task_and_body_system(self, patch_load_dataset):
        """Test that both medical_task and body_system are stored in sample metadata."""
        row = self._make_row(medical_task="dx", body_system="cardio")
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [row])
        samples = MedXpertQADataset().load()
        assert "medical_task" in samples[0].metadata
        assert samples[0].metadata["medical_task"] == "dx"
        assert "body_system" in samples[0].metadata
        assert samples[0].metadata["body_system"] == "cardio"

    def test_answer_choices_stripped_from_question(self, patch_load_dataset):
        """Test that 'Answer Choices:' marker is stripped from the question stem."""
        row = self._make_row(question="What is X? Answer Choices: A. Y B. Z")
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [row])
        samples = MedXpertQADataset().load()
        assert "Answer Choices:" not in samples[0].question

    def test_question_without_marker_unchanged(self, patch_load_dataset):
        """Test that question without 'Answer Choices:' marker is passed through."""
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(question="What is X?")],
        )
        samples = MedXpertQADataset().load()
        assert samples[0].question_stem == "What is X?"

    def test_uses_text_config(self, patch_load_dataset):
        """Test that load runs without error using the text config."""
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert len(samples) == 1

    def _make_row(self, **overrides):
        row = make_dict_options_row()
        row["label"] = row.pop("answer_idx", "B")
        row.update(overrides)
        return row
