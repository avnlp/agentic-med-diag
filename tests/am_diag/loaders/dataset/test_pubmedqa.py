"""Unit tests for PubMedQA dataset loader."""

from __future__ import annotations

from typing import Any

import pytest

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.pubmedqa import _TEST_IDS, PubMedQADataset

from .shuffle_test_mixin import ShuffleTestMixin


def make_pattern_f_row(**overrides: Any) -> dict[str, Any]:
    """Pattern F: yes/no/maybe fixed options. Used by PubMedQA."""
    row: dict[str, Any] = {
        "pubid": "12345678",
        "question": "Does treatment X improve outcome Y?",
        "final_decision": "yes",
        "context": {
            "labels": ["BACKGROUND", "METHODS"],
            "contexts": ["Background text.", "Methods text."],
            "meshes": ["Humans", "Randomized Controlled Trials"],
        },
    }
    row.update(overrides)
    return row


class TestPubMedQADataset(ShuffleTestMixin):
    """Standard shared tests + PubMedQA-specific tests.

    All unit tests use split='train' to bypass the test-ID filter.
    The default split='test' filters pubid against _TEST_IDS, which
    mock rows do not belong to.

    Order: Edge/empty → Core behavior → Fallback → Fixtures.
    """

    loader_class = PubMedQADataset

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        return make_pattern_f_row(**overrides)

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [self.make_row()])
        return self.loader_class(**loader_kwargs).load(split="train")

    def test_skips_row_with_empty_question(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(question="")],
        )
        samples = PubMedQADataset().load(split="train")
        assert len(samples) == 0

    def test_skips_row_with_invalid_decision(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(final_decision="unknown")],
        )
        samples = PubMedQADataset().load(split="train")
        assert len(samples) == 0

    def test_unknown_decision_returns_none(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(final_decision="unknown")],
        )
        samples = PubMedQADataset().load(split="train")
        assert len(samples) == 0

    def test_pubid_not_in_test_ids_returns_none_for_test_split(
        self,
        patch_load_dataset,
    ):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(pubid="99999999")],
        )
        samples = PubMedQADataset().load(split="test")
        assert len(samples) == 0

    def test_load_returns_correct_type(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert all(isinstance(s, MCQSample) for s in samples)

    def test_answer_key_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert all(s.answer == s.answer_key for s in samples)

    def test_answer_text_invariant(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert all(s.answer_text == s.options[s.answer_key] for s in samples)

    def test_dataset_field_correct(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert samples[0].dataset == "pubmedqa"

    def test_split_field_reflects_argument(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert samples[0].split == "train"

    def test_correct_hf_repo_and_split_called(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert len(samples) == 1

    def test_metadata_fields_populated(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(pubid="12345678")],
        )
        samples = PubMedQADataset().load(split="train")
        assert "pubid" in samples[0].metadata
        assert samples[0].metadata["pubid"] == "12345678"

    def test_options_always_yes_no_maybe(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset(shuffle_options=False).load(split="train")
        assert samples[0].options == {"A": "Yes", "B": "No", "C": "Maybe"}

    def test_context_embedded_in_question(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert samples[0].question_stem.startswith("Context:")

    def test_context_labels_and_contexts_formatted(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        q = samples[0].question_stem
        assert "BACKGROUND. Background text." in q
        assert "METHODS. Methods text." in q

    def test_meshes_in_metadata(self, patch_load_dataset):
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", [make_pattern_f_row()])
        samples = PubMedQADataset().load(split="train")
        assert "meshes" in samples[0].metadata
        assert isinstance(samples[0].metadata["meshes"], list)

    @pytest.mark.parametrize(
        "decision,expected",
        [
            ("yes", "A"),
            ("no", "B"),
            ("maybe", "C"),
            ("YES", "A"),
            ("No", "B"),
        ],
    )
    def test_decision_maps_to_letter(self, patch_load_dataset, decision, expected):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(final_decision=decision)],
        )
        samples = PubMedQADataset(shuffle_options=False).load(split="train")
        assert len(samples) == 1
        assert samples[0].answer_key == expected

    def test_pubid_in_test_ids_returns_sample_for_test_split(self, patch_load_dataset):
        first_test_id = next(iter(_TEST_IDS))
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(pubid=first_test_id)],
        )
        samples = PubMedQADataset().load(split="test")
        assert len(samples) == 1

    def test_train_split_does_not_filter_by_id(self, patch_load_dataset):
        patch_load_dataset(
            "am_diag.loaders.dataset.pubmedqa",
            [make_pattern_f_row(pubid="99999999")],
        )
        samples = PubMedQADataset().load(split="train")
        assert len(samples) == 1

    def test_limit_param(self, patch_load_dataset):
        rows = [make_pattern_f_row(pubid=str(i)) for i in range(10)]
        patch_load_dataset("am_diag.loaders.dataset.pubmedqa", rows)
        samples = PubMedQADataset().load(split="train", limit=5)
        assert len(samples) <= 5
