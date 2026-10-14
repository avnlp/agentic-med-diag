"""Multiple-choice question sample model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, computed_field, model_validator


@dataclass(frozen=True, slots=True)
class ExtractedMCQ:
    """The seam between a loader's per-row parse and `MCQDataset._build_sample`.

    A normalized, pre-shuffle view of one row. `answer_keys` is always a list
    (length 1 for single-answer). `is_multi_answer` is declared by the loader,
    never inferred from `len(answer_keys)`. `sample_id` defaults to `None`, in
    which case the base falls back to `str(idx)`.

    Attributes:
        question_stem: The raw question text (without answer choices).
        options: Mapping of option labels to option texts.
        answer_keys: Correct option labels (length 1 for single-answer).
        is_multi_answer: Whether the question expects multiple correct answers.
        metadata: Dataset-specific provenance for the sample.
        sample_id: Stable id for the row, or `None` to fall back to `str(idx)`.
        row_id: Stable row number mixed into the shuffle seed, or `None` to use
            the iteration index.
    """

    question_stem: str
    options: dict[str, str]
    answer_keys: list[str]
    is_multi_answer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    row_id: int | None = None


class MCQSample(BaseModel):
    """A single normalized MCQ sample.

    Stores only independent state -- ``options``, ``answer_keys``, and
    ``is_multi_answer``. The singular answer accessors are derived, so the
    following invariants hold structurally rather than by assertion:

    - ``answer == answer_key == answer_keys[0]``
    - ``answer_text == answer_texts[0] == options[answer_keys[0]]``
    - ``answer_texts[i] == options[answer_keys[i]]`` for all ``i``

    ``is_multi_answer`` is declared by the loader, never inferred from the
    number of correct answers: a select-all question with a single correct
    answer is ``answer_keys=["A"], is_multi_answer=True``.

    When ``shuffle_options`` is enabled, ``options``/``answer_keys`` reflect the
    shuffled state and ``options_original`` preserves the pre-shuffle order.
    """

    sample_id: str
    question: str
    question_stem: str
    options: dict[str, str]
    options_original: dict[str, str] | None = None
    answer_keys: list[str]
    is_multi_answer: bool = False
    dataset: str
    split: str
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def _check_answer_keys(self) -> MCQSample:
        """Reject empty, duplicate, or unknown ``answer_keys``.

        Multiple keys require ``is_multi_answer=True``.
        """
        if not self.answer_keys:
            raise ValueError("answer_keys must be non-empty")
        if len(set(self.answer_keys)) != len(self.answer_keys):
            raise ValueError(f"answer_keys {self.answer_keys} contain duplicates")
        if len(self.answer_keys) > 1 and not self.is_multi_answer:
            raise ValueError("multiple answer_keys require is_multi_answer=True")
        missing = [k for k in self.answer_keys if k not in self.options]
        if missing:
            raise ValueError(
                f"answer_keys {missing} absent from options {list(self.options)}",
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answer_key(self) -> str:
        """The primary correct option label."""
        return self.answer_keys[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answer(self) -> str:
        """Alias of ``answer_key`` for the shared ``QASample`` shape."""
        return self.answer_keys[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answer_text(self) -> str:
        """Text of the primary correct option."""
        return self.options[self.answer_keys[0]]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answer_texts(self) -> list[str]:
        """Texts of all correct options, parallel to ``answer_keys``."""
        return [self.options[k] for k in self.answer_keys]
