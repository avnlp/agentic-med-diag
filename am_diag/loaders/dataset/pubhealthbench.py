"""PubHealthBench dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


class PubHealthBenchDataset(QADataset):
    """Loader for PubHealthBench dataset.

    Args:
        category: Filter to a specific question category. Defaults to `None`
            (all categories).
    """

    dataset_name: ClassVar[str] = "pubhealthbench"
    hf_repo: ClassVar[str] = "Joshua-Harris/PubHealthBench"
    default_split: ClassVar[str] = "reviewed"

    def __init__(
        self,
        category: str | None = None,
        shuffle_options: bool = True,
        shuffle_seed: int | None = 1618,
    ) -> None:
        """Initialise the PubHealthBench MCQ dataset loader.

        Args:
            category: Filter to a specific question category.
            shuffle_options: Randomly permute MCQ option order.
            shuffle_seed: Seed for deterministic shuffling.
        """
        super().__init__(shuffle_options=shuffle_options, shuffle_seed=shuffle_seed)
        self._category = category

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the PubHealthBench MCQ dataset.

        Args:
            split: Dataset split (`"test"`, `"reviewed"`, etc.).
            limit: Maximum number of samples.

        Returns:
            List of MCQSample instances.
        """
        target_split = split or self.default_split
        ds = load_dataset(self.hf_repo, split=target_split, streaming=True)
        if self._category is not None:
            cat = self._category
            ds = ds.filter(lambda r: r.get("category") == cat)
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
        answer_index = row.get("answer_index")
        if answer_index is None:
            return None
        try:
            answer_key = chr(ord("A") + int(answer_index))
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
            sample_id=str(row.get("question_id", idx)),
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
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "source": row.get("source", ""),
            },
        )
