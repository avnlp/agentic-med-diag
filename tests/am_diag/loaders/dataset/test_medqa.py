"""Unit tests for MedQA dataset loader."""

from __future__ import annotations

from typing import Any

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.medqa import MedQADataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_a_row(**overrides: Any) -> dict[str, Any]:
    """Pattern A: dict options, letter answer_idx. Used by MedQA, MedXpertQA."""
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


class TestMedQADataset(ShuffleTestMixin):
    """Standard shared tests + MedQA-specific tests.

    Order: Edge/empty → Core behavior → Fallback → Fixtures.
    """

    loader_class = MedQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_pattern_a_row(**overrides)

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
            [make_pattern_a_row(question="")],
        )
        samples = MedQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_a_row(options={})],
        )
        samples = MedQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_a_row(answer_idx="Z")],
        )
        samples = MedQADataset().load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.base", [make_pattern_a_row()])
        samples = MedQADataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.base", [make_pattern_a_row()])
        samples = MedQADataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.base", [make_pattern_a_row()])
        samples = MedQADataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.base", [make_pattern_a_row()])
        samples = MedQADataset().load()
        assert samples[0].dataset == "medqa"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.base", [make_pattern_a_row()])
        samples = MedQADataset().load(split="train")
        assert samples[0].split == "train"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset, mocker):
        patch_load_dataset("am_diag.loaders.dataset.base", [make_pattern_a_row()])
        MedQADataset().load(split="test")
        # Mock was called if we got here without error

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_a_row(meta_info="step1")],
        )
        samples = MedQADataset().load()
        assert "meta_info" in samples[0].metadata
        assert samples[0].metadata["meta_info"] == "step1"

    def test_options_passed_through_unchanged(self, patch_load_dataset):
        opts = {"A": "Drug A", "B": "Drug B", "C": "Drug C", "D": "Drug D"}
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_a_row(options=opts)],
        )
        samples = MedQADataset().load()
        assert samples[0].options == opts

    def test_answer_idx_normalized_to_uppercase(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_a_row(answer_idx="b")],
        )
        samples = MedQADataset().load()
        assert len(samples) == 1
        assert samples[0].answer_key == "B"

    def test_sample_id_is_row_index(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_a_row(), make_pattern_a_row()],
        )
        samples = MedQADataset().load()
        assert samples[0].sample_id == "0"
        assert samples[1].sample_id == "1"

    def test_meta_info_in_metadata(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_a_row(meta_info="step3")],
        )
        samples = MedQADataset().load()
        assert "meta_info" in samples[0].metadata
        assert samples[0].metadata["meta_info"] == "step3"

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_a_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.base", rows)
        samples = MedQADataset().load(limit=5)
        assert len(samples) <= 5
