"""MMLU-Pro Health dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


_VALID_ANSWER_KEYS: frozenset[str] = frozenset("ABCDEFGHIJ")


class MMLUProHealthDataset(QADataset):
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
        """Load the MMLU-Pro Health dataset.

        Args:
            split: Dataset split (`"test"` or other splits).
            limit: Maximum number of samples.

        Returns:
            List of MCQSample instances.
        """
        target_split = split or self.default_split

        # Populate few-shot examples from validation.
        self.few_shot_examples = []
        val_ds = load_dataset(self.hf_repo, split="validation", streaming=True)
        val_ds = val_ds.filter(lambda r: (r.get("category") or "").lower() == "health")
        for idx, row in enumerate(val_ds):
            sample = self._row_to_sample(idx, row, "validation")
            if sample is not None:
                self.few_shot_examples.append(sample)
            if len(self.few_shot_examples) >= self._num_few_shot:
                break

        ds = load_dataset(self.hf_repo, split=target_split, streaming=True)
        ds = ds.filter(lambda r: (r.get("category") or "").lower() == "health")
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
        options = {
            chr(ord("A") + i): str(v)
            for i, v in enumerate(raw_options)
            if str(v).strip().upper() != "N/A"
        }
        if not options:
            return None
        answer_key = str(row.get("answer") or "").strip().upper()
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
                "category": row.get("category", ""),
                "cot_content": row.get("cot_content", ""),
                "src": row.get("src", ""),
                "question_id": row.get("question_id", ""),
            },
        )
