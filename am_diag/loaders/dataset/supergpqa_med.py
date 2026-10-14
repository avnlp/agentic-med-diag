"""SuperGPQA Medicine dataset loader."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


_VALID_ANSWER_KEYS: frozenset[str] = frozenset("ABCDEFGHIJ")


class SuperGPQAMedDataset(MCQDataset):
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

    def _rows(self, split: str) -> Iterable[Any]:
        ds = load_dataset(self.hf_repo, split=split, streaming=True)
        ds = ds.filter(lambda r: (r.get("discipline") or "") == "Medicine")
        if self._field is not None:
            fld = self._field
            ds = ds.filter(lambda r: r.get("field") == fld)
        if self._difficulty is not None:
            diff = self._difficulty
            ds = ds.filter(lambda r: r.get("difficulty") == diff)
        yield from ds

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        stem = (record.get("question") or "").strip()
        raw_options: list[Any] = record.get("options") or []
        if not stem or not raw_options:
            return None
        options = {chr(ord("A") + i): str(v) for i, v in enumerate(raw_options)}
        key = str(record.get("answer_letter") or "").strip().upper()
        if key not in _VALID_ANSWER_KEYS or key not in options:
            return None
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={
                "field": record.get("field", ""),
                "difficulty": record.get("difficulty", ""),
                "discipline": record.get("discipline", ""),
                "source": record.get("source", ""),
            },
        )
