"""SuperGPQA Medicine dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


_VALID_ANSWER_KEYS: frozenset[str] = frozenset("ABCDEFGHIJ")


class SuperGPQAMedDataset(QADataset):
    """Loader for the SuperGPQA Medicine dataset.

    Args:
        field: Restrict to a specific medical field. Defaults to `None` (all).
        difficulty: Restrict to a specific difficulty level. Defaults to `None`.
    """

    dataset_name: ClassVar[str] = "supergpqa_med"
    hf_repo: ClassVar[str] = "m-a-p/SuperGPQA"
    default_split: ClassVar[str] = "train"

    def __init__(
        self,
        field: str | None = None,
        difficulty: str | None = None,
        shuffle_options: bool = True,
        shuffle_seed: int | None = 1618,
    ) -> None:
        """Initialise the SuperGPQA Medicine dataset loader.

        Args:
            field: Restrict to a specific medical field.
            difficulty: Restrict to a difficulty level.
            shuffle_options: Randomly permute MCQ option order.
            shuffle_seed: Seed for deterministic shuffling.
        """
        super().__init__(shuffle_options=shuffle_options, shuffle_seed=shuffle_seed)
        self._field = field
        self._difficulty = difficulty

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the SuperGPQA Medicine dataset.

        Args:
            split: Dataset split (defaults to `"train"`).
            limit: Maximum number of samples.

        Returns:
            List of MCQSample instances.
        """
        target_split = split or self.default_split
        ds = load_dataset(self.hf_repo, split=target_split, streaming=True)
        ds = ds.filter(lambda r: (r.get("discipline") or "") == "Medicine")
        if self._field is not None:
            fld = self._field
            ds = ds.filter(lambda r: r.get("field") == fld)
        if self._difficulty is not None:
            diff = self._difficulty
            ds = ds.filter(lambda r: r.get("difficulty") == diff)
        if limit is not None:
            ds = ds.take(limit)
        result: list[QASample] = []
        for idx, row in enumerate(ds):
            sample = self._row_to_sample(idx, row, target_split)
            if sample is not None:
                result.append(sample)
        return result

    def _row_to_sample(self, idx: int, row: Any, split: str) -> MCQSample | None:
        question_stem = (row.get("question") or "").strip()
        if not question_stem:
            return None
        raw_options: list[Any] = row.get("options") or []
        if not raw_options:
            return None
        options = {chr(ord("A") + i): str(v) for i, v in enumerate(raw_options)}
        answer_key = str(row.get("answer_letter") or "").strip().upper()
        if answer_key not in _VALID_ANSWER_KEYS or answer_key not in options:
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
            sample_id=str(idx),
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
                "field": row.get("field", ""),
                "difficulty": row.get("difficulty", ""),
                "discipline": row.get("discipline", ""),
                "source": row.get("source", ""),
            },
        )
