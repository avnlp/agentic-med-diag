"""MMLU-Pro Health dataset loader."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


_VALID_ANSWER_KEYS: frozenset[str] = frozenset("ABCDEFGHIJ")


def _is_health(row: Any) -> bool:
    """Whether a raw MMLU-Pro row belongs to the health category."""
    return (row.get("category") or "").lower() == "health"


class MMLUProHealthDataset(MCQDataset):
    """Loader for the MMLU-Pro Health dataset."""

    dataset_name: ClassVar[str] = "mmlu_pro_health"
    hf_repo: ClassVar[str] = "TIGER-Lab/MMLU-Pro"
    default_split: ClassVar[str] = "test"

    def __init__(
        self,
        num_few_shot: int = 5,
        shuffle_options: bool = True,
        shuffle_seed: int | None = 1618,
    ) -> None:
        """Initialise the MMLU-Pro Health dataset loader.

        Args:
            num_few_shot: Number of validation examples for few-shot.
            shuffle_options: Randomly permute MCQ option order.
            shuffle_seed: Seed for deterministic shuffling.
        """
        super().__init__(shuffle_options=shuffle_options, shuffle_seed=shuffle_seed)
        self._num_few_shot = num_few_shot
        self.few_shot_examples: list[MCQSample] = []

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the dataset, first populating `few_shot_examples` from validation.

        Args:
            split: Dataset split to load. Defaults to `self.default_split`.
            limit: Maximum number of samples to return.

        Returns:
            List of parsed `MCQSample` instances (a `QASample` subtype).
        """
        self.few_shot_examples = self._collect_few_shot()
        return super().load(split, limit)

    def _collect_few_shot(self) -> list[MCQSample]:
        """Build up to `num_few_shot` examples from the validation split."""
        ds = load_dataset(self.hf_repo, split="validation", streaming=True)
        ds = ds.filter(_is_health)
        examples: list[MCQSample] = []
        for idx, row in enumerate(ds):
            if len(examples) >= self._num_few_shot:
                break
            extracted = self._extract(idx, row)
            if extracted is not None:
                examples.append(self._build_sample(idx, extracted, "validation"))
        return examples

    def _rows(self, split: str) -> Iterable[Any]:
        ds = load_dataset(self.hf_repo, split=split, streaming=True)
        yield from ds.filter(_is_health)

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        stem = (record.get("question") or "").strip()
        if not stem:
            return None
        raw_options: list[Any] = record.get("options") or []
        options = {
            chr(ord("A") + i): str(v)
            for i, v in enumerate(raw_options)
            if str(v).strip().upper() != "N/A"
        }
        if not options:
            return None
        key = str(record.get("answer") or "").strip().upper()
        if key not in _VALID_ANSWER_KEYS or key not in options:
            return None
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={
                "category": record.get("category", ""),
                "cot_content": record.get("cot_content", ""),
                "src": record.get("src", ""),
                "question_id": record.get("question_id", ""),
            },
            sample_id=str(record.get("question_id", idx)),
        )
