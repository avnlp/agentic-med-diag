"""Abstract base class for Question Answer dataset loaders.

Loaders with custom split-mapping logic, multiple HF configs, or additional filtering
override `load` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import ExtractedMCQ, MCQSample, QASample
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


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


class MCQDataset(QADataset):
    """Deep base for multiple-choice (MCQ) dataset loaders.

    Owns the full MCQ pipeline as a template method: iterate rows -> extract ->
    shuffle options -> enforce invariants -> format the prompt -> construct
    `MCQSample`. Subclasses fill two holes:

    - `_rows(split)`: iteration. The default loads a single HF config via
      streaming; override it for multiple configs, subject loops, or filtering,
      yielding records that may carry per-config provenance (e.g.
      `(subject, row)` tuples).
    - `_extract(idx, record)`: a pure per-row parse to `ExtractedMCQ`, or `None`
      to skip the row.

    No MCQ loader overrides `load()`; the shared shuffle-and-build logic lives
    once in `_build_sample`.
    """

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the dataset, returning a list of `MCQSample` instances.

        Args:
            split: Dataset split to load. Defaults to `self.default_split`.
            limit: Maximum number of samples to return (counted after skips).

        Returns:
            List of parsed `MCQSample` instances (a `QASample` subtype).
        """
        target_split = split or self.default_split
        out: list[QASample] = []
        for idx, record in enumerate(self._rows(target_split)):
            if limit is not None and len(out) >= limit:
                break
            extracted = self._extract(idx, record)
            if extracted is not None:
                out.append(self._build_sample(idx, extracted, target_split))
        return out

    def _rows(self, split: str) -> Iterable[Any]:
        """Yield raw records for the split.

        The default loads a single HF config via streaming. Override for
        multiple configs, subject loops, or row filtering.

        Args:
            split: The active split name.

        Yields:
            Raw records (HF row dicts, or tuples carrying provenance).
        """
        args = (
            (self.hf_repo,)
            if self.hf_config is None
            else (self.hf_repo, self.hf_config)
        )
        yield from load_dataset(*args, split=split, streaming=True)

    @abstractmethod
    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        """Parse one record into an `ExtractedMCQ`, or `None` to skip it.

        Args:
            idx: Row index (used as fallback sample id).
            record: A record yielded by `_rows`.

        Returns:
            An `ExtractedMCQ`, or `None` to skip this record.
        """
        raise NotImplementedError

    def _build_sample(
        self,
        idx: int,
        extracted: ExtractedMCQ,
        split: str,
    ) -> MCQSample:
        """Shuffle, format, and construct the `MCQSample` for one extracted row.

        Args:
            idx: Row index, mixed into the shuffle seed and used as fallback id.
            extracted: The normalized, pre-shuffle row.
            split: The requested split, recorded on the sample.

        Returns:
            The constructed `MCQSample`.
        """
        options = extracted.options
        answer_keys = extracted.answer_keys
        options_original: dict[str, str] | None = None
        if self.shuffle_options:
            options_original = options
            options, answer_keys, _ = shuffle_options(
                options,
                answer_keys,
                seed=self.shuffle_seed,
                row_id=idx,
            )
        return MCQSample(
            sample_id=extracted.sample_id or str(idx),
            question=format_mcq_prompt(
                extracted.question_stem,
                options,
                is_multi_answer=extracted.is_multi_answer,
            ),
            question_stem=extracted.question_stem,
            options=options,
            options_original=options_original,
            answer_keys=answer_keys,
            is_multi_answer=extracted.is_multi_answer,
            dataset=self.dataset_name,
            split=split,
            metadata=extracted.metadata,
        )
