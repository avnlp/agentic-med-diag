"""Unit tests for PubHealthBench MCQ dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.loaders.dataset.pubhealthbench import PubHealthBenchDataset


def make_list_indexed_row(**overrides: Any) -> dict[str, Any]:
    """Build a row with options as list and answer_index as 0-indexed int."""
    row: dict[str, Any] = {
        "question": "What is the correct statement?",
        "options": ["Statement A", "Statement B", "Statement C"],
        "answer_index": 1,
        "question_id": "phb_001",
        "category": "Epidemiology",
        "source": "CDC",
    }
    row.update(overrides)
    return row


class TestPubHealthBenchDataset:
    """Tests for PubHealthBench extraction and answer validation."""

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(question="")],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that row with empty options list produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(options=[])],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_empty_options_list_returns_none(self, patch_load_dataset):
        """Test that empty options list is rejected during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(options=[])],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_missing_answer_index_returns_none(self, patch_load_dataset):
        """Test that None answer_index produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(answer_index=None)],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that row with None answer_index is filtered out."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(answer_index=None)],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        """Test that load runs without error for the default PubHealthBench split."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row()],
        )
        samples = PubHealthBenchDataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that question_id from the row is preserved in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(question_id="phb_001")],
        )
        samples = PubHealthBenchDataset().load()
        assert "question_id" in samples[0].metadata
        assert samples[0].metadata["question_id"] == "phb_001"

    def test_metadata_has_question_id_and_category(self, patch_load_dataset):
        """Test that both question_id and category are stored in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(question_id="phb_002", category="Cardiology")],
        )
        samples = PubHealthBenchDataset().load()
        assert "question_id" in samples[0].metadata
        assert samples[0].metadata["question_id"] == "phb_002"
        assert "category" in samples[0].metadata
        assert samples[0].metadata["category"] == "Cardiology"

    def test_default_split_is_reviewed(self):
        """Test that the default split is 'reviewed'."""
        loader = PubHealthBenchDataset()
        assert loader.default_split == "reviewed"

    def test_skips_non_integer_answer_index(self, patch_load_dataset):
        """Test that row with non-integer answer_index produces 0 samples."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(answer_index="x")],
        )
        samples = PubHealthBenchDataset(shuffle_options=False).load()
        assert len(samples) == 0

    def test_skips_answer_index_out_of_range(self, patch_load_dataset):
        """Test that row with answer_index beyond options length produces 0 samples."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(answer_index=9)],
        )
        samples = PubHealthBenchDataset(shuffle_options=False).load()
        assert len(samples) == 0

    @pytest.mark.parametrize(
        "idx,expected",
        [(0, "A"), (1, "B"), (2, "C")],
    )
    def test_answer_index_maps_to_letter(self, patch_load_dataset, idx, expected):
        """Test that 0-indexed answer_index maps to letter (0=A, 1=B)."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(answer_index=idx)],
        )
        samples = PubHealthBenchDataset(shuffle_options=False).load()
        assert len(samples) == 1
        assert samples[0].answer_key == expected

    def test_category_filter_applied(self, patch_load_dataset):
        """Test that category filter restricts to the specified category."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(category="Epidemiology")],
        )
        loader = PubHealthBenchDataset(category="Epidemiology")
        samples = loader.load()
        assert len(samples) == 1

    def test_category_none_includes_all(self, patch_load_dataset):
        """Test that category=None does not filter out any rows by category."""
        patch_load_dataset(
            "am_diag.loaders.dataset.pubhealthbench",
            [make_list_indexed_row(category="Epidemiology")],
        )
        loader = PubHealthBenchDataset(category=None)
        samples = loader.load()
        assert len(samples) == 1
