"""PubHealthBench dataset loader."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


class PubHealthBenchDataset(MCQDataset):
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

    def _rows(self, split: str) -> Iterable[Any]:
        ds = load_dataset(self.hf_repo, split=split, streaming=True)
        if self._category is not None:
            cat = self._category
            ds = ds.filter(lambda r: r.get("category") == cat)
        yield from ds

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        stem = (record.get("question") or "").strip()
        raw_options: list[Any] = record.get("options") or []
        if not stem or not raw_options:
            return None
        options = {chr(ord("A") + i): str(v) for i, v in enumerate(raw_options)}
        answer_index = record.get("answer_index")
        if answer_index is None:
            return None
        try:
            key = chr(ord("A") + int(answer_index))
        except (ValueError, TypeError):
            return None
        if key not in options:
            return None
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={
                "question_id": record.get("question_id", ""),
                "category": record.get("category", ""),
                "source": record.get("source", ""),
            },
            sample_id=str(record.get("question_id", idx)),
        )
