"""Parametrized contract tests for dataset loader .load().

Each param tuple carries a loader class, the patch target for load_dataset,
a valid row dict, the expected dataset name, and optional extras (split,
additional mocks). This file replaces the scattered copies of
test_load_returns_correct_type, test_limit_param, test_dataset_field_correct,
and test_split_field_reflects_argument that used to live in every per-loader
test file.

Tests are organised into two families:

- ``TestMCQLoadContract`` — 10 MCQ loaders, all returning ``list[MCQSample]``.
- ``TestFreeFormLoadContract`` — 6 free-form loaders, returning various
  ``QASample`` subtypes.
"""

from __future__ import annotations

import pytest

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.careqa import CareQADataset
from am_diag.loaders.dataset.careqa_reasoning import CareQAReasoningDataset
from am_diag.loaders.dataset.healthbench import HealthBenchDataset
from am_diag.loaders.dataset.medcase_reasoning import MedCaseReasoningDataset
from am_diag.loaders.dataset.medmcqa import MedMCQADataset
from am_diag.loaders.dataset.medqa import MedQADataset
from am_diag.loaders.dataset.medxpertqa import MedXpertQADataset
from am_diag.loaders.dataset.mmlu_med import MMLUMedDataset
from am_diag.loaders.dataset.mmlu_pro_health import MMLUProHealthDataset
from am_diag.loaders.dataset.nejm_diagnostic import NEJMDiagnosticDataset
from am_diag.loaders.dataset.nejm_qa import NEJMQADataset
from am_diag.loaders.dataset.pubhealthbench import PubHealthBenchDataset
from am_diag.loaders.dataset.pubhealthbench_freeform import (
    PubHealthBenchFreeformDataset,
)
from am_diag.loaders.dataset.pubmedqa import PubMedQADataset
from am_diag.loaders.dataset.rar_med import RARMedDataset
from am_diag.loaders.dataset.supergpqa_med import SuperGPQAMedDataset


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NEJM_QA_ROW = {
    "question": (
        "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid"
    ),
    "answer": "B",
}
_MMLU_PRO_ROW = {
    "question": "Which intervention?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "B",
    "answer_letter": "B",
    "category": "health",
    "cot_content": "",
    "src": "pubmed",
    "question_id": "q001",
}
_PUBMEDQA_ROW = {
    "pubid": "12345678",
    "question": "Does X improve Y?",
    "final_decision": "yes",
    "context": {
        "labels": ["BACKGROUND"],
        "contexts": ["Background text."],
    },
}
_MEDXPRTQA_ROW = {
    "question": "Which drug?",
    "options": {"A": "Drug A", "B": "Drug B", "C": "Drug C", "D": "Drug D"},
    "label": "B",
    "meta_info": "step2",
}
_HEALTHBENCH_ROW = {
    "prompt_id": "hb_001",
    "prompt": [{"role": "user", "content": "Hello."}],
    "rubrics": [],
    "example_tags": ["theme:test"],
}

# ---------------------------------------------------------------------------
# MCQ loader contract tests
# ---------------------------------------------------------------------------


class TestMCQLoadContract:
    """Contract tests shared by every MCQ dataset loader.

    Each test proves correct return type, dataset name, limit, and split.
    Verified per-loader to catch wiring errors.
    """

    @pytest.mark.parametrize(
        "loader_cls,patch_target,row,expected_dataset,split,kwargs,extra_mocks",
        [
            pytest.param(
                MedQADataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "Which drug?",
                    "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                    "answer_idx": "B",
                },
                "medqa",
                "test",
                {"shuffle_options": False},
                None,
                id="medqa",
            ),
            pytest.param(
                CareQADataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "What is first-line?",
                    "op1": "A",
                    "op2": "B",
                    "op3": "C",
                    "op4": "D",
                    "cop": 3,
                },
                "careqa",
                "test",
                {"shuffle_options": False},
                None,
                id="careqa",
            ),
            pytest.param(
                MedMCQADataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "Which drug?",
                    "opa": "A",
                    "opb": "B",
                    "opc": "C",
                    "opd": "D",
                    "cop": 2,
                    "subject_name": "Pharma",
                },
                "medmcqa",
                "test",
                {"shuffle_options": False},
                None,
                id="medmcqa",
            ),
            pytest.param(
                MMLUMedDataset,
                "am_diag.loaders.dataset.mmlu_med",
                {
                    "question": "Which is correct?",
                    "choices": ["A", "B", "C", "D"],
                    "answer": 2,
                },
                "mmlu_med",
                "test",
                {"subjects": ["anatomy"], "shuffle_options": False},
                None,
                id="mmlu_med",
            ),
            pytest.param(
                MMLUProHealthDataset,
                "am_diag.loaders.dataset.mmlu_pro_health",
                _MMLU_PRO_ROW,
                "mmlu_pro_health",
                "test",
                {"shuffle_options": False, "num_few_shot": 0},
                None,
                id="mmlu_pro_health",
            ),
            pytest.param(
                NEJMQADataset,
                "am_diag.loaders.dataset.nejm_qa",
                _NEJM_QA_ROW,
                "nejm_qa",
                "all",
                {"shuffle_options": False},
                {"patch_nejm_specialties": True},
                id="nejm_qa",
            ),
            pytest.param(
                PubHealthBenchDataset,
                "am_diag.loaders.dataset.pubhealthbench",
                {
                    "question": "What is correct?",
                    "options": ["A", "B", "C"],
                    "answer_index": 1,
                    "category": "Epidemiology",
                },
                "pubhealthbench",
                "reviewed",
                {"shuffle_options": False},
                None,
                id="pubhealthbench",
            ),
            pytest.param(
                PubMedQADataset,
                "am_diag.loaders.dataset.pubmedqa",
                _PUBMEDQA_ROW,
                "pubmedqa",
                "train",
                {"shuffle_options": False},
                None,
                id="pubmedqa",
            ),
            pytest.param(
                MedXpertQADataset,
                "am_diag.loaders.dataset.medxpertqa",
                _MEDXPRTQA_ROW,
                "medxpertqa",
                "test",
                {"shuffle_options": False},
                None,
                id="medxpertqa",
            ),
            pytest.param(
                SuperGPQAMedDataset,
                "am_diag.loaders.dataset.supergpqa_med",
                {
                    "question": "Which intervention?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "B",
                    "answer_letter": "B",
                    "discipline": "Medicine",
                    "category": "health",
                },
                "supergpqa_med",
                "train",
                {"shuffle_options": False},
                None,
                id="supergpqa_med",
            ),
        ],
    )
    def test_returns_mcq_samples(
        self,
        loader_cls,
        patch_target,
        row,
        expected_dataset,
        split,
        kwargs,
        extra_mocks,
        patch_load_dataset,
        mocker,
    ):
        """Test that the loader returns list[MCQSample] with correct properties."""
        if extra_mocks and extra_mocks.get("patch_nejm_specialties"):
            mocker.patch(
                "am_diag.loaders.dataset.nejm_qa._SPECIALTIES",
                ["pediatrics"],
            )
        patch_load_dataset(patch_target, [row])
        loader = loader_cls(**kwargs)
        samples = loader.load(split=split)
        assert len(samples) == 1
        assert isinstance(samples[0], MCQSample)
        assert samples[0].dataset == expected_dataset
        assert samples[0].split == split

    @pytest.mark.parametrize(
        "loader_cls,patch_target,row,kwargs,split,extra_mocks",
        [
            pytest.param(
                MedQADataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "Which drug?",
                    "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                    "answer_idx": "B",
                },
                {"shuffle_options": False},
                "test",
                None,
                id="medqa",
            ),
            pytest.param(
                CareQADataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "What is first-line?",
                    "op1": "A",
                    "op2": "B",
                    "op3": "C",
                    "op4": "D",
                    "cop": 3,
                },
                {"shuffle_options": False},
                "test",
                None,
                id="careqa",
            ),
            pytest.param(
                MedMCQADataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "Which drug?",
                    "opa": "A",
                    "opb": "B",
                    "opc": "C",
                    "opd": "D",
                    "cop": 2,
                    "subject_name": "Pharma",
                },
                {"shuffle_options": False},
                "test",
                None,
                id="medmcqa",
            ),
            pytest.param(
                MMLUMedDataset,
                "am_diag.loaders.dataset.mmlu_med",
                {
                    "question": "Which is correct?",
                    "choices": ["A", "B", "C", "D"],
                    "answer": 2,
                },
                {"subjects": ["anatomy"], "shuffle_options": False},
                "test",
                None,
                id="mmlu_med",
            ),
            pytest.param(
                MMLUProHealthDataset,
                "am_diag.loaders.dataset.mmlu_pro_health",
                _MMLU_PRO_ROW,
                {"shuffle_options": False, "num_few_shot": 0},
                "test",
                None,
                id="mmlu_pro_health",
            ),
            pytest.param(
                NEJMQADataset,
                "am_diag.loaders.dataset.nejm_qa",
                _NEJM_QA_ROW,
                {"shuffle_options": False},
                "all",
                {"patch_nejm_specialties": True},
                id="nejm_qa",
            ),
            pytest.param(
                PubHealthBenchDataset,
                "am_diag.loaders.dataset.pubhealthbench",
                {
                    "question": "What is correct?",
                    "options": ["A", "B", "C"],
                    "answer_index": 1,
                },
                {"shuffle_options": False},
                "reviewed",
                None,
                id="pubhealthbench",
            ),
            pytest.param(
                PubMedQADataset,
                "am_diag.loaders.dataset.pubmedqa",
                _PUBMEDQA_ROW,
                {"shuffle_options": False},
                "train",
                None,
                id="pubmedqa",
            ),
            pytest.param(
                MedXpertQADataset,
                "am_diag.loaders.dataset.medxpertqa",
                _MEDXPRTQA_ROW,
                {"shuffle_options": False},
                "test",
                None,
                id="medxpertqa",
            ),
            pytest.param(
                SuperGPQAMedDataset,
                "am_diag.loaders.dataset.supergpqa_med",
                {
                    "question": "Which intervention?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "B",
                    "answer_letter": "B",
                    "discipline": "Medicine",
                },
                {"shuffle_options": False},
                "train",
                None,
                id="supergpqa_med",
            ),
        ],
    )
    def test_limit_caps_count(
        self,
        loader_cls,
        patch_target,
        row,
        kwargs,
        split,
        extra_mocks,
        patch_load_dataset,
        mocker,
    ):
        """Test that limit restricts the number of samples returned."""
        if extra_mocks and extra_mocks.get("patch_nejm_specialties"):
            mocker.patch(
                "am_diag.loaders.dataset.nejm_qa._SPECIALTIES",
                ["pediatrics"],
            )
        rows = [dict(row) for _ in range(10)]
        patch_load_dataset(patch_target, rows)
        loader = loader_cls(**kwargs)
        samples = loader.load(split=split, limit=3)
        assert len(samples) <= 3

    def test_sample_id_present(self, patch_load_dataset):
        """Test that all MCQ samples have a non-empty sample_id."""
        row = {
            "question": "Which drug?",
            "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "answer_idx": "B",
        }
        patch_load_dataset("am_diag.loaders.dataset.base", [row])
        samples = MedQADataset(shuffle_options=False).load()
        assert samples[0].sample_id is not None
        assert len(samples[0].sample_id) > 0


# ---------------------------------------------------------------------------
# Free-form loader contract tests
# ---------------------------------------------------------------------------


class TestFreeFormLoadContract:
    """Contract tests shared by every free-form dataset loader.

    Free-form loaders return ``QASample`` subtypes.
    ``HealthBenchDataset`` is excluded from generic ``dataset`` / ``split``
    assertions because it sets ``variant`` and ``answer=""``.
    """

    @pytest.mark.parametrize(
        "loader_cls,patch_target,row,expected_dataset,split,kwargs",
        [
            pytest.param(
                CareQAReasoningDataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "What is the treatment?",
                    "answer": "Drug A",
                    "answer_explanation": "Because...",
                },
                "careqa_reasoning",
                "test",
                {},
                id="careqa_reasoning",
            ),
            pytest.param(
                MedCaseReasoningDataset,
                "am_diag.loaders.dataset.base",
                {
                    "case_prompt": "A 45yo with chest pain",
                    "final_diagnosis": "PE",
                    "pmcid": "PMC123",
                },
                "medcase_reasoning",
                "val",
                {},
                id="medcase_reasoning",
            ),
            pytest.param(
                NEJMDiagnosticDataset,
                "am_diag.loaders.dataset.nejm_diagnostic",
                {
                    "case_presentation": "A 55yo with pain",
                    "diagnosis": "Lung Cancer",
                },
                "nejm_diagnostic",
                "nejm_test",
                {},
                id="nejm_diagnostic",
            ),
            pytest.param(
                PubHealthBenchFreeformDataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "What is correct?",
                    "options": ["A", "B", "C"],
                    "answer_index": 1,
                },
                "pubhealthbench_freeform",
                "test",
                {},
                id="pubhealthbench_freeform",
            ),
            pytest.param(
                RARMedDataset,
                "am_diag.loaders.dataset.base",
                {
                    "question": "What is X?",
                    "reference_answer": "Diagnosis X",
                    "rubric": [],
                },
                "rar_med",
                "test",
                {},
                id="rar_med",
            ),
        ],
    )
    def test_returns_qasamples(
        self,
        loader_cls,
        patch_target,
        row,
        expected_dataset,
        split,
        kwargs,
        patch_load_dataset,
    ):
        """Test that the loader returns list[QASample] with correct dataset."""
        patch_load_dataset(patch_target, [row])
        loader = loader_cls(**kwargs)
        samples = loader.load(split=split)
        assert len(samples) == 1
        assert isinstance(samples[0], QASample)
        assert samples[0].dataset == expected_dataset

    @pytest.mark.parametrize(
        "loader_cls,patch_target,row,split,kwargs",
        [
            pytest.param(
                CareQAReasoningDataset,
                "am_diag.loaders.dataset.base",
                {"question": "Q?", "answer": "A", "answer_explanation": "E"},
                "test",
                {},
                id="careqa_reasoning",
            ),
            pytest.param(
                MedCaseReasoningDataset,
                "am_diag.loaders.dataset.base",
                {
                    "case_prompt": "A 45yo with pain",
                    "final_diagnosis": "PE",
                    "pmcid": "PMC123",
                },
                "val",
                {},
                id="medcase_reasoning",
            ),
            pytest.param(
                NEJMDiagnosticDataset,
                "am_diag.loaders.dataset.nejm_diagnostic",
                {"case_presentation": "Pain", "diagnosis": "Dx"},
                "nejm_test",
                {},
                id="nejm_diagnostic",
            ),
            pytest.param(
                PubHealthBenchFreeformDataset,
                "am_diag.loaders.dataset.base",
                {"question": "Q?", "options": ["A"], "answer_index": 0},
                "test",
                {},
                id="pubhealthbench_freeform",
            ),
            pytest.param(
                RARMedDataset,
                "am_diag.loaders.dataset.base",
                {"question": "Q?", "reference_answer": "A", "rubric": []},
                "test",
                {},
                id="rar_med",
            ),
            pytest.param(
                HealthBenchDataset,
                "am_diag.loaders.dataset.healthbench",
                _HEALTHBENCH_ROW,
                "test",
                {"variant": "all"},
                id="healthbench",
            ),
        ],
    )
    def test_freeform_limit_caps_count(
        self,
        loader_cls,
        patch_target,
        row,
        split,
        kwargs,
        patch_load_dataset,
    ):
        """Test that limit restricts the number of free-form samples returned."""
        rows = [dict(row) for _ in range(10)]
        patch_load_dataset(patch_target, rows)
        loader = loader_cls(**kwargs)
        samples = loader.load(split=split, limit=3)
        assert len(samples) <= 3
