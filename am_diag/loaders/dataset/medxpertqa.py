"""MedXpertQA dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


class MedXpertQADataset(QADataset):
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

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the MedXpertQA dataset.

        Args:
            split: Dataset split (`"test"` or other splits).
            limit: Maximum number of samples.

        Returns:
            List of MCQSample instances.
        """
        target_split = split or self.default_split
        ds = load_dataset(self.hf_repo, "Text", split=target_split, streaming=True)
        if self._question_type is not None:
            qt = self._question_type
            ds = ds.filter(lambda r: (r.get("question_type") or "") == qt)
        if limit is not None:
            ds = ds.take(limit)
        result: list[QASample] = []
        for idx, row in enumerate(ds):
            sample = self._row_to_sample(idx, row, target_split)
            if sample is not None:
                result.append(sample)
        return result

    def _row_to_sample(self, idx: int, row: Any, split: str) -> MCQSample | None:
        raw_question = str(row.get("question") or "")
        if "Answer Choices:" in raw_question:
            question_stem = raw_question.partition("Answer Choices:")[0].strip()
        else:
            question_stem = raw_question.strip()
        if not question_stem:
            return None
        options: dict[str, str] = row.get("options") or {}
        if not options:
            return None
        answer_key = str(row.get("label") or "").strip().upper()
        if answer_key not in options:
            return None
        answer_text = options[answer_key]

        options_original: dict[str, str] | None = None
        if self.shuffle_options:
            shuffled, new_key, _ = shuffle_options(
                options,
                answer_key,
                seed=self.shuffle_seed,
                row_id=idx,
            )
            options_original = options
            options = shuffled
            answer_key = new_key
            answer_text = options[answer_key]

        question = format_mcq_prompt(question_stem, options)
        return MCQSample(
            sample_id=str(row.get("id", idx)),
            question=question,
            question_stem=question_stem,
            options=options,
            options_original=options_original,
            answer_key=answer_key,
            answer_text=answer_text,
            answer=answer_key,
            dataset=self.dataset_name,
            split=split,
            metadata={
                "medical_task": row.get("medical_task", ""),
                "body_system": row.get("body_system", ""),
                "question_type": row.get("question_type", ""),
                "id": row.get("id", ""),
            },
        )
