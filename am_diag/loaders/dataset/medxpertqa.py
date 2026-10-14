"""MedXpertQA dataset loader."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


class MedXpertQADataset(MCQDataset):
    """Loader for the MedXpertQA multiple-choice question dataset."""

    dataset_name: ClassVar[str] = "medxpertqa"
    hf_repo: ClassVar[str] = "TsinghuaC3I/MedXpertQA"
    default_split: ClassVar[str] = "test"

    def __init__(
        self,
        question_type: str | None = None,
        shuffle_options: bool = True,
        shuffle_seed: int | None = 1618,
    ) -> None:
        """Initialise the MedXpertQA dataset loader.

        Args:
            question_type: Filter to `"reasoning"` or `"understanding"`.
            shuffle_options: Randomly permute MCQ option order.
            shuffle_seed: Seed for deterministic shuffling.
        """
        super().__init__(shuffle_options=shuffle_options, shuffle_seed=shuffle_seed)
        self._question_type = question_type

    def _rows(self, split: str) -> Iterable[Any]:
        ds = load_dataset(self.hf_repo, "Text", split=split, streaming=True)
        if self._question_type is not None:
            qt = self._question_type
            ds = ds.filter(lambda r: (r.get("question_type") or "") == qt)
        yield from ds

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        raw_question = str(record.get("question") or "")
        if "Answer Choices:" in raw_question:
            stem = raw_question.partition("Answer Choices:")[0].strip()
        else:
            stem = raw_question.strip()
        if not stem:
            return None
        options: dict[str, str] = record.get("options") or {}
        key = str(record.get("label") or "").strip().upper()
        if not options or key not in options:
            return None
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={
                "medical_task": record.get("medical_task", ""),
                "body_system": record.get("body_system", ""),
                "question_type": record.get("question_type", ""),
                "id": record.get("id", ""),
            },
            sample_id=str(record.get("id", idx)),
        )
