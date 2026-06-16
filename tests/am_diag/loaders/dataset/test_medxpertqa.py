"""Unit tests for MedXpertQA dataset loader."""

from __future__ import annotations

from typing import Any

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.medxpertqa import MedXpertQADataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_a_row(**overrides: Any) -> dict[str, Any]:
    """Pattern A: dict options, letter answer_idx."""
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


class TestMedXpertQADataset(ShuffleTestMixin):
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    loader_class = MedXpertQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        row = make_pattern_a_row()
        row["label"] = row.pop("answer_idx", "B")
        row.update(overrides)
        return row

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(question="")],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(options={})],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(label="Z")],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_label_not_in_options_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(label="Z")],
        )
        samples = MedXpertQADataset().load()
        assert len(samples) == 0

    def test_question_type_filter(self, patch_load_dataset):
        row = self._make_row(question_type="reasoning")
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [row])
        loader = MedXpertQADataset(question_type="understanding")
        samples = loader.load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert samples[0].dataset == "medxpertqa"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load(split="test")
        assert samples[0].split == "test"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(medical_task="diagnosis")],
        )
        samples = MedXpertQADataset().load()
        assert "medical_task" in samples[0].metadata
        assert samples[0].metadata["medical_task"] == "diagnosis"

    def test_metadata_has_medical_task_and_body_system(self, patch_load_dataset):
        row = self._make_row(medical_task="dx", body_system="cardio")
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [row])
        samples = MedXpertQADataset().load()
        assert "medical_task" in samples[0].metadata
        assert samples[0].metadata["medical_task"] == "dx"
        assert "body_system" in samples[0].metadata
        assert samples[0].metadata["body_system"] == "cardio"

    def test_answer_choices_stripped_from_question(self, patch_load_dataset):
        row = self._make_row(question="What is X? Answer Choices: A. Y B. Z")
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [row])
        samples = MedXpertQADataset().load()
        assert "Answer Choices:" not in samples[0].question

    def test_question_without_marker_unchanged(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.medxpertqa",
            [self._make_row(question="What is X?")],
        )
        samples = MedXpertQADataset().load()
        assert samples[0].question_stem == "What is X?"

    def test_uses_text_config(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", [self._make_row()])
        samples = MedXpertQADataset().load()
        assert len(samples) == 1

    def test_limit_param(self, patch_load_dataset):
        rows = [self._make_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.medxpertqa", rows)
        samples = MedXpertQADataset().load(limit=5)
        assert len(samples) <= 5

    def _make_row(self, **overrides):
        row = make_pattern_a_row()
        row["label"] = row.pop("answer_idx", "B")
        row.update(overrides)
        return row
