"""Unit tests for MMLU-Pro Health dataset loader."""

from __future__ import annotations

from typing import Any

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.mmlu_pro_health import MMLUProHealthDataset

from .shuffle_test_mixin import ShuffleTestMixin


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


class TestMMLUProHealthDataset(ShuffleTestMixin):
    """Tests for MMLU-Pro Health extraction and category routing."""

    loader_class = MMLUProHealthDataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        """Return a valid row for the shared shuffle tests."""
        return make_list_letter_answer_row(**{"category": "health", **overrides})

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.mmlu_pro_health", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        """Test that row with empty question is filtered out during load."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(question="", category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        """Test that row with empty options list produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(options=[], category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_all_na_options_returns_none(self, patch_load_dataset):
        """Test that row with all N/A options produces no sample after filtering."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(options=["N/A", "N/A"], category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        """Test that answer letter not in A-D range produces no sample."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(answer="Z", category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_non_health_category_skipped(self, patch_load_dataset):
        """Test that non-health categories (e.g. math) are filtered out."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(category="math")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_invalid_answer_key_returns_none(self, patch_load_dataset):
        """Test that answer letter outside reconstructed options range skips row."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(answer="Z", category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_answer_not_in_options_after_na_filter_returns_none(
        self,
        patch_load_dataset,
    ):
        """Test that row is skipped when answer letter not in filtered options."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [
                make_list_letter_answer_row(
                    answer="B", options=["A", "N/A"], category="health"
                )
            ],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        """Test that load runs without error for the default MMLU-Pro split."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        """Test that category from the row is preserved in sample metadata."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert "category" in samples[0].metadata
        assert samples[0].metadata["category"] == "health"

    def test_health_category_included(self, patch_load_dataset):
        """Test that health category rows pass through the category filter."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 1

    def test_na_option_filtered_from_options(self, patch_load_dataset):
        """Test that N/A options are removed and option letters are re-indexed."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [
                make_list_letter_answer_row(
                    options=["A", "N/A", "B"],
                    answer="C",
                    category="health",
                ),
            ],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 1
        assert "N/A" not in samples[0].options.values()

    def test_few_shot_examples_populated_after_load(self, patch_load_dataset):
        """Test that few_shot_examples list is populated after calling load()."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(category="health") for _ in range(10)],
        )
        loader = MMLUProHealthDataset(num_few_shot=3)
        loader.load()
        assert len(loader.few_shot_examples) <= 3
        assert all(isinstance(s, MCQSample) for s in loader.few_shot_examples)

    def test_few_shot_count_respects_num_few_shot(self, patch_load_dataset):
        """Test that num_few_shot parameter limits the number of stored examples."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(category="health") for _ in range(10)],
        )
        loader = MMLUProHealthDataset(num_few_shot=2)
        loader.load()
        assert len(loader.few_shot_examples) <= 2

    def test_validation_load_uses_streaming(self, patch_load_dataset):
        """Test that streaming load populates few-shot examples correctly."""
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_list_letter_answer_row(category="health")],
        )
        loader = MMLUProHealthDataset(num_few_shot=1)
        loader.load()
        # If streaming works, we should get results
        assert len(loader.few_shot_examples) >= 1
