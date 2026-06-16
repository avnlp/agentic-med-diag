"""Abstract base class for Question Answer dataset loaders.

Loaders with custom split-mapping logic, multiple HF configs, or additional filtering
override `load` directly.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import QASample


class QADataset(ABC):
    """Abstract base for Question Answer dataset loaders."""

    dataset_name: ClassVar[str]
    hf_repo: ClassVar[str]
    default_split: ClassVar[str] = "test"
    hf_config: ClassVar[str | None] = None

    def __init__(
        self,
        shuffle_options: bool = True,
        shuffle_seed: int | None = 1618,
    ) -> None:
        """Configure MCQ option shuffling for positional bias elimination.

        Args:
        shuffle_options: When True (default), randomly permute MCQ option order to
            eliminate positional bias in accuracy measurements.
        shuffle_seed: Seed for deterministic shuffling.
            - `None`: disables shuffling regardless of `shuffle_options`
            - `-1`: uses non-deterministic random
            - `>= 0`: uses deterministic seed combined with `row_id`.
        """
        self.shuffle_options = shuffle_options
        self.shuffle_seed = shuffle_seed

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the dataset, returning a list of QASample instances.

        Override this method for datasets that need split-mapping, multiple configs, or
        extra filtering logic.

        Args:
            split: Dataset split to load. Defaults to `self.default_split`.
            limit: Maximum number of samples to load.

        Returns:
            List of parsed QASample instances.
        """
        target_split = split or self.default_split
        args = (
            (self.hf_repo,)
            if self.hf_config is None
            else (self.hf_repo, self.hf_config)
        )
        ds = load_dataset(*args, split=target_split, streaming=True)
        if limit is not None:
            ds = ds.take(limit)
        return [
            s
            for idx, row in enumerate(ds)
            if (s := self._row_to_sample(idx, row, target_split)) is not None
        ]

    def _row_to_sample(self, idx: int, row: Any, split: str) -> QASample | None:
        """Parse one dataset row into a `QASample`.

        Args:
            idx: Row index (used as fallback sample ID).
            row: Raw dataset row dict.
            split: The active split name.

        Returns:
            A `QASample` instance, or `None` to skip this row.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _row_to_sample() or override load()",
        )

    def __repr__(self) -> str:
        """Return a string representation of this dataset loader."""
        return f"{type(self).__name__}(dataset={self.dataset_name!r})"
