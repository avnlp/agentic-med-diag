"""Deterministic MCQ grading helpers."""

from __future__ import annotations

import re
import unicodedata

from am_diag.common.data_models import MCQAccuracyResult, MCQSample, MetricResult


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_OPTION_RE = re.compile(r"\b([A-Z])\s*[:.)-]?\s*$")
_LETTER_RE = re.compile(r"\b([A-Z])\b")


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
    answer_text_map = _answer_text_map(sample)
    cleaned_lower = cleaned.lower()

    if cleaned.upper() == sample.answer_key:
        return MCQAccuracyResult(
            True,
            "exact_letter",
            sample.answer_key,
            sample.answer_key,
        )

    if cleaned_lower == answer_text_map[sample.answer_key]:
        return MCQAccuracyResult(
            True,
            "exact_answer_text",
            sample.answer_key,
            sample.answer_key,
        )

    match = _OPTION_RE.search(cleaned.upper())
    if match and match.group(1) == sample.answer_key:
        return MCQAccuracyResult(
            True,
            "terminal_option",
            sample.answer_key,
            sample.answer_key,
        )

    last_letter = _LETTER_RE.findall(cleaned.upper())
    if last_letter and last_letter[-1] == sample.answer_key:
        return MCQAccuracyResult(
            True,
            "tail_letter",
            sample.answer_key,
            sample.answer_key,
        )

    return MCQAccuracyResult(
        False,
        "no_match",
        last_letter[-1] if last_letter else None,
        sample.answer_key,
    )


def _score_multi(sample: MCQSample, prediction: str) -> MCQAccuracyResult:
    cleaned = _clean_prediction(prediction)
    predicted = sorted(dict.fromkeys(_extract_multi_answer(cleaned)))
    correct = sorted(dict.fromkeys(sample.answer_keys))
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
