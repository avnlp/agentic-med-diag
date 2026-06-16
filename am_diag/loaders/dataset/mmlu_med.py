"""MMLU-Med dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


_SUBJECTS: list[str] = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]


class MMLUMedDataset(QADataset):
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

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the MMLU-Med dataset.

        Args:
            split: Dataset split (`"test"` or other splits).
            limit: Maximum number of samples.

        Returns:
            List of MCQSample instances.
        """
        target_split = split or self.default_split
        result: list[QASample] = []
        for subject in self._subjects:
            if limit is not None and len(result) >= limit:
                break
            remaining = limit - len(result) if limit is not None else None
            ds = load_dataset(self.hf_repo, subject, split=target_split, streaming=True)
            if remaining is not None:
                ds = ds.take(remaining)
            for idx, row in enumerate(ds):
                sample = self._row_to_sample(idx, row, target_split, subject=subject)
                if sample is not None:
                    result.append(sample)
        return result

    def _row_to_sample(
        self,
        idx: int,
        row: Any,
        split: str,
        subject: str = "",
    ) -> MCQSample | None:
        question_stem = (row.get("question") or "").strip()
        if not question_stem:
            return None
        choices: list[Any] = row.get("choices") or []
        if not choices:
            return None
        options = {chr(ord("A") + i): str(v) for i, v in enumerate(choices)}
        raw_answer = row.get("answer")
        if raw_answer is None:
            return None
        try:
            answer_key = chr(ord("A") + int(raw_answer))
        except (ValueError, TypeError):
            return None
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
            sample_id=f"{subject}_{idx}",
            question=question,
            question_stem=question_stem,
            options=options,
            options_original=options_original,
            answer_key=answer_key,
            answer_text=answer_text,
            answer=answer_key,
            dataset=self.dataset_name,
            split=split,
            metadata={"subject": subject},
        )
