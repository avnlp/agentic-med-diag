"""Deterministic MCQ grading helpers."""

from __future__ import annotations

import re
import unicodedata

from am_diag.common.data_models import MCQAccuracyResult, MCQSample, MetricResult


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_MARKER_RE = re.compile(
    r"(?i:\b(?:final\s+)?(?:answer|option|choice))\s*(?:is|:|=)?\s*[(\[]?([A-Z])\b"
)
_FINAL_TOKEN_RE = re.compile(r"(?:^|\s)[(\[]?([A-Z])[)\].:]?\s*$")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(text.split())


def _clean_prediction(prediction: str) -> str:
    cleaned = _THINK_RE.sub(" ", prediction)
    cleaned = _ANSWER_RE.sub(r"\1", cleaned)
    cleaned = _BOXED_RE.sub(r"\1", cleaned)
    return _normalize(cleaned)


def _extract_multi_answer(text: str) -> list[str]:
    letters = re.findall(r"\b[A-Z]\b", text.upper())
    if "," in text or ";" in text or " and " in text.lower():
        return sorted(dict.fromkeys(letters))
    return letters[-1:] if letters else []


def _answer_text_map(sample: MCQSample) -> dict[str, str]:
    return {
        key.upper(): _normalize(text).lower() for key, text in sample.options.items()
    }


def _score_single(sample: MCQSample, prediction: str) -> MCQAccuracyResult:
    cleaned = _clean_prediction(prediction)
    answer_key = sample.answer_key.upper()
    answer_text = _answer_text_map(sample)[answer_key]

    if cleaned.upper() == answer_key:
        return MCQAccuracyResult(True, "exact_letter", answer_key, answer_key)

    if cleaned.lower() == answer_text:
        return MCQAccuracyResult(True, "exact_answer_text", answer_key, answer_key)

    markers = _MARKER_RE.findall(cleaned)
    if markers:
        return MCQAccuracyResult(
            markers[-1] == answer_key,
            "answer_marker",
            markers[-1],
            answer_key,
        )

    final_token = _FINAL_TOKEN_RE.search(cleaned)
    if final_token:
        return MCQAccuracyResult(
            final_token.group(1) == answer_key,
            "final_token",
            final_token.group(1),
            answer_key,
        )

    return MCQAccuracyResult(False, "no_match", None, answer_key)


def _score_multi(sample: MCQSample, prediction: str) -> MCQAccuracyResult:
    cleaned = _clean_prediction(prediction)
    predicted = sorted(dict.fromkeys(_extract_multi_answer(cleaned)))
    correct = sorted({key.upper() for key in sample.answer_keys})
    is_correct = predicted == correct
    return MCQAccuracyResult(
        is_correct,
        "multi_answer_set",
        ",".join(predicted) if predicted else None,
        ",".join(correct),
    )


class MCQAccuracyEvaluator:
    """Grade multiple-choice outputs deterministically."""

    metric_name = "mcq_accuracy"

    async def evaluate(self, sample: MCQSample, prediction: str) -> MetricResult:
        """Score a prediction against the sample's answer key."""
        result = (
            _score_multi(sample, prediction)
            if sample.is_multi_answer
            else _score_single(sample, prediction)
        )
        return MetricResult(
            metric_name=self.metric_name,
            score=1.0 if result.correct else 0.0,
            passed=result.correct,
            details={
                "method": result.method,
                "matched_answer": result.matched_answer,
                "correct_answer": result.correct_answer,
                "raw_prediction": prediction,
            },
        )
