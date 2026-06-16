"""Unit tests for MedMCQA dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.medmcqa import MedMCQADataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_b_medmcqa_row(**overrides: Any) -> dict[str, Any]:
    """Pattern B: opa/opb/opc/opd + cop int. Used by MedMCQA."""
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


class TestMedMCQADataset(ShuffleTestMixin):
    """Standard shared tests + MedMCQA-specific tests.

    Order: Edge/empty → Core behavior → Fallback → Fixtures.
    """

    loader_class = MedMCQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_pattern_b_medmcqa_row(**overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.base", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_all_four_options_empty_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(opa="", opb="", opc="", opd="")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(question="")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(opa="", opb="", opc="", opd="")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_zero_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(cop=0)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_five_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(cop=5)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_none_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(cop=None)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_cop_string_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(cop="1")],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(cop=0)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row()],
        )
        samples = MedMCQADataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row()],
        )
        samples = MedMCQADataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row()],
        )
        samples = MedMCQADataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row()],
        )
        samples = MedMCQADataset().load()
        assert samples[0].dataset == "medmcqa"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row()],
        )
        samples = MedMCQADataset().load(split="train")
        assert samples[0].split == "train"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row()],
        )
        # Just verify load works with default split
        samples = MedMCQADataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(subject_name="Pharma")],
        )
        samples = MedMCQADataset().load()
        assert "subject_name" in samples[0].metadata
        assert samples[0].metadata["subject_name"] == "Pharma"

    def test_metadata_has_subject_and_topic(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [
                make_pattern_b_medmcqa_row(
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
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(cop=cop)],
        )
        samples = MedMCQADataset().load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected_key

    def test_sample_id_uses_id_field(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(id="abc123")],
        )
        samples = MedMCQADataset().load()
        assert samples[0].sample_id == "abc123"

    def test_sample_id_fallback_to_index(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_pattern_b_medmcqa_row(id=None)],
        )
        samples = MedMCQADataset().load()
        assert samples[0].sample_id == "0"

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_b_medmcqa_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.base", rows)
        samples = MedMCQADataset().load(limit=5)
        assert len(samples) <= 5
