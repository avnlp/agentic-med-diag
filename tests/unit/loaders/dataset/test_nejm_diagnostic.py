"""Unit tests for NEJM Diagnostic Reasoning open-ended dataset loader."""

from __future__ import annotations

import pytest

from am_diag.loaders.dataset.nejm_diagnostic import NEJMDiagnosticDataset


class TestNEJMDiagnosticDataset:
    """Tests for NEJM Diagnostic extraction and fallback behavior."""

    def test_empty_case_presentation_returns_none(self, patch_load_dataset):
        """Test that row with empty case_presentation is filtered out."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_diagnostic",
            [self._make_row(case_presentation="")],
        )
        samples = NEJMDiagnosticDataset().load()
        assert len(samples) == 0

    def test_title_maps_to_question(self, patch_load_dataset):
        """Test that the Title field is mapped to the sample question."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_diagnostic",
            [self._make_row_with_title(Title="A 55yo male")],
        )
        samples = NEJMDiagnosticDataset().load()
        assert samples[0].question == "A 55yo male"

    def test_answer_maps_to_reference_and_answer(self, patch_load_dataset):
        """Test that the Answer field populates both reference_answer and answer."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_diagnostic",
            [self._make_row_with_title(Answer="Lung Cancer")],
        )
        samples = NEJMDiagnosticDataset().load()
        assert samples[0].reference_answer == "Lung Cancer"
        assert samples[0].answer == "Lung Cancer"

    def test_lab_data_maps_to_reasoning_chain(self, patch_load_dataset):
        """Test that the 'Lab Data' field is mapped to reasoning_chain."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_diagnostic",
            [self._make_row_with_title(**{"Lab Data": "Lab results text"})],
        )
        samples = NEJMDiagnosticDataset().load()
        assert samples[0].reasoning_chain == "Lab results text"

    def test_reasoning_chain_none_when_absent(self, patch_load_dataset):
        """Test that reasoning_chain is None when no lab data is provided."""
        patch_load_dataset(
            "am_diag.loaders.dataset.nejm_diagnostic",
            [self._make_row_with_title_only()],
        )
        samples = NEJMDiagnosticDataset().load()
        assert samples[0].reasoning_chain is None

    def test_default_split_is_nejm_test(self):
        """Test that the default split is 'nejm_test'."""
        assert NEJMDiagnosticDataset.default_split == "nejm_test"

    def test_limit_caps_loaded_samples(self, patch_load_dataset):
        """Test that load(limit=N) returns at most N samples."""
        rows = [self._make_row_with_title() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.nejm_diagnostic", rows)
        samples = NEJMDiagnosticDataset().load(limit=5)
        assert len(samples) == 5

    def test_unknown_config_raises(self):
        """Test that unknown split config raises ValueError."""
        with pytest.raises(ValueError, match="Unknown config"):
            NEJMDiagnosticDataset().load(split="bogus")

    def _make_row(self, **overrides):
        row = {
            "case_presentation": "A 55yo male presents with...",
            "diagnosis": "Lung Cancer",
            "discussion": "The patient was diagnosed because...",
            "specialty": "Oncology",
            "source": "NEJM",
        }
        row.update(overrides)
        return row

    def _make_row_with_title(self, **overrides):
        row = {
            "Title": "A 55yo male presents with chest pain",
            "Answer": "Myocardial Infarction",
            "Lab Data": "ECG shows ST elevation",
        }
        row.update(overrides)
        return row

    def _make_row_with_title_only(self, **overrides):
        row = {
            "Title": "A 55yo male presents with chest pain",
            "Answer": "Myocardial Infarction",
        }
        row.update(overrides)
        return row
