"""NEJM AI Q&A MCQ dataset loader."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


_SPECIALTIES: list[str] = [
    "general_surgery",
    "internal_medicine",
    "psychiatry",
    "pediatrics",
    "obgyn",
]

_OPTIONS_RE = re.compile(r"(?:\n|^)([A-D])\.\s*(.+)")


class NEJMQADataset(QADataset):
    r"""Loader for the NEJM AI Q&A dataset."""

    dataset_name: ClassVar[str] = "nejm_qa"
    hf_repo: ClassVar[str] = "nejm-ai-qa/exams"
    default_split: ClassVar[str] = "all"

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load NEJM AI Q&A exam questions.

        The dataset has a single `default` config with 5 specialty
        splits. Passing split='all' (default) loads all specialties.

        Args:
            split: `"all"` (default) or a specialty split name
                (`general_surgery`, `internal_medicine`, etc.).
            limit: Maximum number of samples.

        Returns:
            List of MCQSample instances.
        """
        # The dataset has no train/test splits; instead it has 5 specialty configs.
        target_split = split or self.default_split
        if target_split == "all":
            splits_to_load = _SPECIALTIES
        elif target_split in _SPECIALTIES:
            splits_to_load = [target_split]
        else:
            msg = f"Unknown split '{target_split}'. Use 'all' or one of {_SPECIALTIES}"
            raise ValueError(msg)

        result: list[QASample] = []
        for split_name in splits_to_load:
            ds = load_dataset(self.hf_repo, "default", split=split_name, streaming=True)
            if limit is not None:
                ds = ds.take(limit)
            for idx, row in enumerate(ds):
                sample = self._row_to_sample(idx, row, split_name)  # type: ignore[arg-type]
                if sample is not None:
                    result.append(sample)
                if limit is not None and len(result) >= limit:
                    break
            if limit is not None and len(result) >= limit:
                break
        return result

    def _extract_stem_and_options(self, text: str) -> tuple[str, dict[str, str]] | None:
        matches = list(_OPTIONS_RE.finditer(text))
        if not matches:
            return None
        stem = text[: matches[0].start()].strip()
        if not stem:
            return None
        options = {m.group(1): m.group(2).strip() for m in matches}
        # Require at least 2 options to be a valid MCQ
        if len(options) < 2:
            return None
        return stem, options

    def _row_to_sample(self, idx: int, row: Any, split: str) -> MCQSample | None:
        q_text = (row.get("question") or "").strip()
        if not q_text:
            return None

        parsed = self._extract_stem_and_options(q_text)
        if parsed is None:
            return None
        question_stem, options = parsed

        answer_raw = str(row.get("answer") or "").strip()
        if not answer_raw:
            return None

        answer_letters = [a.strip() for a in answer_raw.split(",") if a.strip()]
        valid_letters = [a for a in answer_letters if a in options]
        if not valid_letters:
            return None

        is_multi = len(valid_letters) > 1

        answer_keys = list(valid_letters)
        answer_key = answer_keys[0]
        answer_texts = [options[k] for k in answer_keys]
        answer_text = answer_texts[0]

        options_original: dict[str, str] | None = None
        if self.shuffle_options:
            options_original = options
            if is_multi:
                shuffled, answer_keys, _ = shuffle_options(
                    options,
                    answer_keys,
                    seed=self.shuffle_seed,
                    row_id=idx,
                )
                options = shuffled
                answer_key = answer_keys[0]
                answer_texts = [options[k] for k in answer_keys]
                answer_text = answer_texts[0]
            else:
                shuffled, answer_key, _ = shuffle_options(
                    options,
                    answer_key,
                    seed=self.shuffle_seed,
                    row_id=idx,
                )
                options = shuffled
                answer_text = options[answer_key]
                answer_keys = [answer_key]
                answer_texts = [answer_text]

        question = format_mcq_prompt(question_stem, options, is_multi_answer=is_multi)
        return MCQSample(
            sample_id=f"{split}_{idx}",
            question=question,
            question_stem=question_stem,
            options=options,
            options_original=options_original,
            answer_key=answer_key,
            answer_text=answer_text,
            answer=answer_key,
            answer_keys=answer_keys,
            answer_texts=answer_texts,
            is_multi_answer=is_multi,
            dataset=self.dataset_name,
            split=split,
            metadata={
                "specialty": split,
                "answer_raw": answer_raw,
                "answer_letters": answer_letters,
            },
        )
