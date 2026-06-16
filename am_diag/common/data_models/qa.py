"""Open-ended QA sample model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class OpenEndedSample(BaseModel):
    """A free-form open-ended QA sample."""

    sample_id: str
    question: str
    question_stem: str = ""
    reference_answer: str
    reasoning_chain: str | None = None
    dataset: str
    split: str = "test"
    answer: str
    metadata: dict[str, Any]
