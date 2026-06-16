"""MedQA dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


class MedQADataset(QADataset):
    """Loader for the MedQA multiple-choice question dataset."""

    dataset_name: ClassVar[str] = "medqa"
    hf_repo: ClassVar[str] = "GBaker/MedQA-USMLE-4-options"
    default_split: ClassVar[str] = "test"

    def _row_to_sample(self, idx: int, row: Any, split: str) -> MCQSample | None:
        question_stem = (row.get("question") or "").strip()
        if not question_stem:
            return None
        options: dict[str, str] = row.get("options") or {}
        if not options:
            return None
        answer_key = str(row.get("answer_idx") or "").strip().upper()
        if answer_key not in options:
            return None
        answer_text = options[answer_key]

        options_original: dict[str, str] | None = None
        if self.shuffle_options:
            shuffled, new_key, _ = shuffle_options(
                options,
                answer_key,
                seed=self.shuffle_seed,
                row_id=idx,
            )
            options_original = options
            options = shuffled
            answer_key = new_key
            answer_text = options[answer_key]

        question = format_mcq_prompt(question_stem, options)
        return MCQSample(
            sample_id=str(idx),
            question=question,
            question_stem=question_stem,
            options=options,
            options_original=options_original,
            answer_key=answer_key,
            answer_text=answer_text,
            answer=answer_key,
            dataset=self.dataset_name,
            split=split,
            metadata={"meta_info": row.get("meta_info", "")},
        )
