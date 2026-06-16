"""Unit tests for RAR-Medicine rubric dataset loader."""

from __future__ import annotations

from am_diag.common.data_models import RARMedSample
from am_diag.loaders.dataset.rar_med import RARMedDataset


class TestRARMedDataset:
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    def test_empty_question_returns_none(self, patch_load_dataset):
        row = {"question": "", "reference_answer": "A", "rubric": []}
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = RARMedDataset().load()
        assert len(samples) == 0

    def test_basic_row_produces_rar_med_sample(self, patch_load_dataset):
        row = {
            "question": "What is the treatment for X?",
            "reference_answer": "Drug A",
            "rubric": [{"criterion": "test", "points": 1}],
        }
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = RARMedDataset().load()
        assert len(samples) == 1
        assert isinstance(samples[0], RARMedSample)

    def test_rubrics_kept_as_raw_list_of_dicts(self, patch_load_dataset):
        row = {
            "question": "Q?",
            "reference_answer": "A",
            "rubric": [{"criterion": "test", "points": 1}],
        }
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = RARMedDataset().load()
        assert isinstance(samples[0].rubrics, list)
        assert isinstance(samples[0].rubrics[0], dict)

    def test_answer_equals_reference_answer(self, patch_load_dataset):
        row = {
            "question": "Q?",
            "reference_answer": "Diagnosis X",
            "rubric": [],
        }
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = RARMedDataset().load()
        assert samples[0].answer == samples[0].reference_answer

    def test_excluded_keys_not_in_metadata(self, patch_load_dataset):
        row = {
            "question": "Q?",
            "reference_answer": "A",
            "rubric": [],
            "extra_field": "value",
        }
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = RARMedDataset().load()
        assert "question" not in samples[0].metadata
        assert "reference_answer" not in samples[0].metadata
        assert "rubric" not in samples[0].metadata

    def test_extra_row_keys_present_in_metadata(self, patch_load_dataset):
        row = {
            "question": "Q?",
            "reference_answer": "A",
            "rubric": [],
            "extra_field": "value",
            "other_col": 42,
        }
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = RARMedDataset().load()
        assert "extra_field" in samples[0].metadata
        assert samples[0].metadata["extra_field"] == "value"
        assert "other_col" in samples[0].metadata
        assert samples[0].metadata["other_col"] == 42

    def test_limit_param(self, patch_load_dataset):
        rows = [
            {"question": f"Q{i}?", "reference_answer": "A", "rubric": []}
            for i in range(10)
        ]
        patch_load_dataset("am_diag.loaders.dataset.base", rows)
        samples = RARMedDataset().load(limit=5)
        assert len(samples) <= 5
