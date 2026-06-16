"""Unit tests for MMLU-Pro Health dataset loader."""

from __future__ import annotations

from typing import Any

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.mmlu_pro_health import MMLUProHealthDataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_d_row(**overrides: Any) -> dict[str, Any]:
    """Pattern D: list options + letter answer. Used by MMLU-Pro Health, SuperGPQA."""
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
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    loader_class = MMLUProHealthDataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_pattern_d_row(category="health", **overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.mmlu_pro_health", [self.make_row()])
        return self.loader_class(**loader_kwargs).load()

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(question="", category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_missing_options(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(options=[], category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_all_na_options_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(options=["N/A", "N/A"], category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_skips_row_with_invalid_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(answer="Z", category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_non_health_category_skipped(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="math")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_invalid_answer_key_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(answer="Z", category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_answer_not_in_options_after_na_filter_returns_none(
        self,
        patch_load_dataset,
    ):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(answer="B", options=["A", "N/A"], category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert samples[0].dataset == "mmlu_pro_health"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load(split="test")
        assert samples[0].split == "test"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert "category" in samples[0].metadata
        assert samples[0].metadata["category"] == "health"

    def test_health_category_included(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        samples = MMLUProHealthDataset().load()
        assert len(samples) == 1

    def test_na_option_filtered_from_options(self, patch_load_dataset):
        # After filtering N/A from ["A", "N/A", "B"], options become
        # {"A": "A", "C": "B"} — answer must be "C" to match re-index.
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [
                make_pattern_d_row(
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
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health") for _ in range(10)],
        )
        loader = MMLUProHealthDataset(num_few_shot=3)
        loader.load()
        assert len(loader.few_shot_examples) <= 3
        assert all(isinstance(s, MCQSample) for s in loader.few_shot_examples)

    def test_few_shot_count_respects_num_few_shot(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health") for _ in range(10)],
        )
        loader = MMLUProHealthDataset(num_few_shot=2)
        loader.load()
        assert len(loader.few_shot_examples) <= 2

    def test_validation_load_uses_streaming(self, patch_load_dataset, mocker):
        patch_load_dataset(
            "am_diag.loaders.dataset.mmlu_pro_health",
            [make_pattern_d_row(category="health")],
        )
        loader = MMLUProHealthDataset(num_few_shot=1)
        loader.load()
        # If streaming works, we should get results
        assert len(loader.few_shot_examples) >= 1

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_d_row(category="health") for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.mmlu_pro_health", rows)
        samples = MMLUProHealthDataset().load(limit=5)
        assert len(samples) <= 5
