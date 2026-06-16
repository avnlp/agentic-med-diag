"""Multiple-choice question sample model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MCQSample(BaseModel):
    """A single MCQ sample normalized.

    Invariants enforced by every MCQ loader:
    - `answer == answer_key` (always an uppercase letter: "A", "B", …)
    - `answer_text == options[answer_key]`
    - `answer_key == answer_keys[0]`
    - `answer_text == answer_texts[0]`
    - `answer_texts[i] == options[answer_keys[i]]` for all i

    For questions with a single correct answer, `answer_keys` is a
    one-element list (e.g. `["A"]`) and `is_multi_answer` is `False`.

    When `shuffle_options` is enabled:
    - `question` contains the combined prompt (question stem + shuffled options).
    - `options_original` preserves the original (unshuffled) option order for audit.
    - `answer_key`, `answer_text`, and `options` reflect the shuffled state.
    """

    sample_id: str
    question: str
    question_stem: str
    options: dict[str, str]
    options_original: dict[str, str] | None = None
    answer_key: str
    answer_text: str
    answer: str
    answer_keys: list[str] = []
    answer_texts: list[str] = []
    is_multi_answer: bool = False
    dataset: str
    split: str
    metadata: dict[str, Any]
