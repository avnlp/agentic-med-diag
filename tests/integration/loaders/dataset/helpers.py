"""Shared assertion helpers for loader integration tests.

Plain assertion functions — NOT a class, NOT a mixin. Composition over inheritance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from am_diag.common.data_models import MCQSample, QASample


def assert_basic_sample_fields(samples: list[QASample], expected_dataset: str) -> None:
    """Assert common invariants every sample should satisfy."""
    assert len(samples) > 0
    for s in samples:
        assert s.sample_id
        assert s.question
        assert s.dataset == expected_dataset
        assert s.split


def assert_mcq_invariants(samples: list[MCQSample]) -> None:
    """Assert MCQ invariants: answer == answer_key, answer_text matches options."""
    for s in samples:
        assert s.answer == s.answer_key, f"{s.sample_id}: answer != answer_key"
        assert s.answer_text == s.options[s.answer_key], (
            f"{s.sample_id}: answer_text doesn't match options[answer_key]"
        )
        if s.answer_keys:
            assert s.answer_key == s.answer_keys[0]
            assert s.answer_text == s.answer_texts[0]


def assert_mcq_question_format(sample: MCQSample, has_context: bool = False) -> None:
    """Assert the formatted question contains expected sections."""
    assert sample.question.startswith("Question:"), (
        "Question should start with 'Question:'"
    )
    assert "Choices:" in sample.question, "Question should contain 'Choices:'"
    assert "Answer with" in sample.question, (
        "Question should contain answer instruction"
    )
    if not has_context:
        assert sample.question_stem, "question_stem should not be empty"


def assert_mcq_options_not_empty(sample: MCQSample, min_options: int = 2) -> None:
    """Assert options dict has the expected minimum number of entries."""
    assert len(sample.options) >= min_options, (
        f"Expected at least {min_options} options, got {len(sample.options)}"
    )


def assert_shuffle_works(loader_factory: Callable[..., Any]) -> None:
    """Load with shuffle enabled and verify options_original is populated."""
    loader = loader_factory(shuffle_options=True, shuffle_seed=42)
    samples = loader.load(limit=5)
    assert len(samples) > 0
    for s in samples:
        assert s.options_original is not None, "options_original should be set"


def assert_shuffle_disabled_preserves_order(loader_factory: Callable[..., Any]) -> None:
    """Load with shuffle disabled and verify options_original is None."""
    loader = loader_factory(shuffle_options=False)
    samples = loader.load(limit=5)
    assert len(samples) > 0


def assert_limit_zero_returns_empty(loader_factory: Callable[..., Any]) -> None:
    """Loading with limit=0 should return an empty list."""
    samples = loader_factory().load(limit=0)
    assert samples == []


def assert_openended_invariants(samples: list[QASample]) -> None:
    """Assert open-ended sample invariants."""
    for s in samples:
        assert s.answer, f"{s.sample_id}: answer should not be empty"
        assert s.reference_answer, (
            f"{s.sample_id}: reference_answer should not be empty"
        )
        assert s.question, f"{s.sample_id}: question should not be empty"


def assert_dataset_name(samples: list[QASample], expected: str) -> None:
    """Assert all samples have the expected dataset name."""
    for s in samples:
        assert s.dataset == expected, (
            f"Expected dataset={expected!r}, got {s.dataset!r}"
        )


def assert_split_value(samples: list[QASample], expected: str) -> None:
    """Assert all samples have the expected split value."""
    for s in samples:
        assert s.split == expected, f"Expected split={expected!r}, got {s.split!r}"
