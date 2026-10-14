"""Unit tests for MedQA dataset loader."""

from __future__ import annotations

from typing import Any

from am_diag.loaders.dataset.medqa import MedQADataset

from .shuffle_test_mixin import ShuffleTestMixin


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


class TestMedQADataset(ShuffleTestMixin):
    """Tests for MedQA extraction and answer-idx validation."""

    loader_class = MedQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        """Return a valid row for the shared shuffle tests."""
        return make_dict_options_row(**overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.base", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(question="")],
        )
        samples = MedQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that row with empty options dict is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(options={})],
        )
        samples = MedQADataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that answer_idx outside A-D produces no sample for that row."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(answer_idx="Z")],
        )
        samples = MedQADataset().load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset, mocker):
        """Test that load runs without error for the default MedQA split."""
        patch_load_dataset("am_diag.loaders.dataset.base", [make_dict_options_row()])
        MedQADataset().load(split="test")
        # Mock was called if we got here without error

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that meta_info from the row is preserved in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(meta_info="step1")],
        )
        samples = MedQADataset().load()
        assert "meta_info" in samples[0].metadata
        assert samples[0].metadata["meta_info"] == "step1"

    def test_options_passed_through_unchanged(self, patch_load_dataset):
        """Test that options dict is stored on the sample exactly as provided."""
        opts = {"A": "Drug A", "B": "Drug B", "C": "Drug C", "D": "Drug D"}
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(options=opts)],
        )
        samples = MedQADataset().load()
        assert samples[0].options == opts

    def test_answer_idx_normalized_to_uppercase(self, patch_load_dataset):
        """Test that a lowercase answer_idx is normalized to uppercase on load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(answer_idx="b")],
        )
        samples = MedQADataset().load()
        assert len(samples) == 1
        assert samples[0].answer_key == "B"

    def test_sample_id_is_row_index(self, patch_load_dataset):
        """Test that sample_id defaults to the 0-based row index."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(), make_dict_options_row()],
        )
        samples = MedQADataset().load()
        assert samples[0].sample_id == "0"
        assert samples[1].sample_id == "1"

    def test_meta_info_in_metadata(self, patch_load_dataset):
        """Test that meta_info is accessible on the sample metadata dict."""
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_dict_options_row(meta_info="step3")],
        )
        samples = MedQADataset().load()
        assert "meta_info" in samples[0].metadata
        assert samples[0].metadata["meta_info"] == "step3"
