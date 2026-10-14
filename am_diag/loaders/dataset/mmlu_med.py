"""MMLU-Med dataset loader."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


_SUBJECTS: list[str] = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]


class MMLUMedDataset(MCQDataset):
    """Loader for the MMLU-Med (6-subject subset) multiple-choice question dataset.

    The included subjects are: anatomy, clinical knowledge, college biology,
    college medicine, medical genetics, and professional medicine.
    """

    dataset_name: ClassVar[str] = "mmlu_med"
    hf_repo: ClassVar[str] = "cais/mmlu"
    default_split: ClassVar[str] = "test"

    def __init__(
        self,
        subjects: list[str] | None = None,
        shuffle_options: bool = True,
        shuffle_seed: int | None = 1618,
    ) -> None:
        """Initialise the MMLU-Med dataset loader.

        Args:
            subjects: Medical subjects to load. Defaults to all six.
            shuffle_options: Randomly permute MCQ option order.
            shuffle_seed: Seed for deterministic shuffling.
        """
        super().__init__(shuffle_options=shuffle_options, shuffle_seed=shuffle_seed)
        self._subjects = subjects if subjects is not None else list(_SUBJECTS)

    def _rows(self, split: str) -> Iterable[tuple[str, int, Any]]:
        for subject in self._subjects:
            ds = load_dataset(self.hf_repo, subject, split=split, streaming=True)
            for row_idx, row in enumerate(ds):
                yield subject, row_idx, row

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        subject, row_idx, row = record
        stem = (row.get("question") or "").strip()
        choices: list[Any] = row.get("choices") or []
        if not stem or not choices:
            return None
        options = {chr(ord("A") + i): str(v) for i, v in enumerate(choices)}
        raw_answer = row.get("answer")
        if raw_answer is None:
            return None
        try:
            key = chr(ord("A") + int(raw_answer))
        except (ValueError, TypeError):
            return None
        if key not in options:
            return None
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={"subject": subject},
            sample_id=f"{subject}_{row_idx}",
            row_id=row_idx,
        )
