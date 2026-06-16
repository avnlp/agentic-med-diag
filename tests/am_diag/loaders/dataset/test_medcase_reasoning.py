"""Unit tests for MedCaseReasoning open-ended dataset loader."""

from __future__ import annotations

from typing import Any

from am_diag.loaders.dataset.medcase_reasoning import MedCaseReasoningDataset


def make_medcase_row(**overrides: Any) -> dict[str, Any]:
    """Row factory for MedCaseReasoning samples."""
    row: dict[str, Any] = {
        "pmcid": "PMC1234567",
        "case_prompt": "A 45-year-old presents with ...",
        "final_diagnosis": "Pulmonary Embolism",
        "diagnostic_reasoning": "Step 1: ... Step 2: ...",
        "title": "Case Report: PE",
        "journal": "NEJM",
        "article_link": "https://...",
        "publication_date": "2023-01-15",
    }
    row.update(overrides)
    return row


class TestMedCaseReasoningDataset:
    """Order: Edge/empty → Core behavior → Fallback → Fixtures."""

    def test_empty_case_prompt_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_medcase_row(case_prompt="")],
        )
        samples = MedCaseReasoningDataset().load()
        assert len(samples) == 0

    def test_case_prompt_maps_to_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_medcase_row(case_prompt="A 45yo with chest pain")],
        )
        samples = MedCaseReasoningDataset().load()
        assert samples[0].question == "A 45yo with chest pain"

    def test_final_diagnosis_maps_to_reference_and_answer(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_medcase_row(final_diagnosis="PE")],
        )
        samples = MedCaseReasoningDataset().load()
        assert samples[0].reference_answer == "PE"
        assert samples[0].answer == "PE"

    def test_diagnostic_reasoning_maps_to_reasoning_chain(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_medcase_row(diagnostic_reasoning="Step 1: ...")],
        )
        samples = MedCaseReasoningDataset().load()
        assert samples[0].reasoning_chain == "Step 1: ..."

    def test_reasoning_chain_none_when_absent(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_medcase_row(diagnostic_reasoning=None)],
        )
        samples = MedCaseReasoningDataset().load()
        assert samples[0].reasoning_chain is None

    def test_sample_id_uses_pmcid(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_medcase_row(pmcid="PMC123")],
        )
        samples = MedCaseReasoningDataset().load()
        assert samples[0].sample_id == "PMC123"

    def test_metadata_has_required_fields(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.base",
            [make_medcase_row()],
        )
        samples = MedCaseReasoningDataset().load()
        meta = samples[0].metadata
        assert "pmcid" in meta
        assert "title" in meta
        assert "journal" in meta
        assert "article_link" in meta
        assert "publication_date" in meta

    def test_default_split_is_val(self):
        loader = MedCaseReasoningDataset()
        assert loader.default_split == "val"

    def test_limit_param(self, patch_load_dataset):
        rows = [make_medcase_row() for _ in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.base", rows)
        samples = MedCaseReasoningDataset().load(limit=5)
        assert len(samples) <= 5
