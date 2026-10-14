"""NEJM AI Q&A MCQ dataset loader."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


_SPECIALTIES: list[str] = [
    "general_surgery",
    "internal_medicine",
    "psychiatry",
    "pediatrics",
    "obgyn",
]

_OPTIONS_RE = re.compile(r"(?:\n|^)([A-D])\.\s*(.+)")


class NEJMQADataset(MCQDataset):
    r"""Loader for the NEJM AI Q&A dataset.

    The dataset has a single `default` config with 5 specialty splits. Passing
    `split="all"` (default) loads all specialties; the per-sample specialty is
    recorded in `metadata`, while `split` reflects the requested split.
    """

    dataset_name: ClassVar[str] = "nejm_qa"
    hf_repo: ClassVar[str] = "nejm-ai-qa/exams"
    default_split: ClassVar[str] = "all"

    def _rows(self, split: str) -> Iterable[tuple[str, int, Any]]:
        if split == "all":
            specialties = _SPECIALTIES
        elif split in _SPECIALTIES:
            specialties = [split]
        else:
            msg = f"Unknown split '{split}'. Use 'all' or one of {_SPECIALTIES}"
            raise ValueError(msg)
        for specialty in specialties:
            ds = load_dataset(self.hf_repo, "default", split=specialty, streaming=True)
            for row_idx, row in enumerate(ds):
                yield specialty, row_idx, row

    def _extract_stem_and_options(
        self,
        text: str,
    ) -> tuple[str, dict[str, str]] | None:
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

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        specialty, row_idx, row = record
        q_text = (row.get("question") or "").strip()
        if not q_text:
            return None

        parsed = self._extract_stem_and_options(q_text)
        if parsed is None:
            return None
        stem, options = parsed

        answer_raw = str(row.get("answer") or "").strip()
        if not answer_raw:
            return None

        answer_letters = [a.strip() for a in answer_raw.split(",") if a.strip()]
        valid_letters = [a for a in answer_letters if a in options]
        if not valid_letters:
            return None

        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=list(valid_letters),
            is_multi_answer=len(valid_letters) > 1,
            metadata={
                "specialty": specialty,
                "answer_raw": answer_raw,
                "answer_letters": answer_letters,
            },
            sample_id=f"{specialty}_{row_idx}",
            row_id=row_idx,
        )
