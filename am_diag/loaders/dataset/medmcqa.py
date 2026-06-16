"""MedMCQA dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


class MedMCQADataset(QADataset):
    """Loader for the MedMCQA multiple-choice question dataset."""

    dataset_name: ClassVar[str] = "medmcqa"
    hf_repo: ClassVar[str] = "lighteval/med_mcqa"
    default_split: ClassVar[str] = "validation"

    def _row_to_sample(self, idx: int, row: Any, split: str) -> MCQSample | None:
        question_stem = (row.get("question") or "").strip()
        if not question_stem:
            return None
        opa = (row.get("opa") or "").strip()
        opb = (row.get("opb") or "").strip()
        opc = (row.get("opc") or "").strip()
        opd = (row.get("opd") or "").strip()
        if not (opa or opb or opc or opd):
            return None
        cop = row.get("cop")
        if cop not in (1, 2, 3, 4):
            return None
        options = {"A": opa, "B": opb, "C": opc, "D": opd}
        answer_key = ["A", "B", "C", "D"][int(cop) - 1]
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
            sample_id=str(row.get("id") or idx),
            question=question,
            question_stem=question_stem,
            options=options,
            options_original=options_original,
            answer_key=answer_key,
            answer_text=answer_text,
            answer=answer_key,
            dataset=self.dataset_name,
            split=split,
            metadata={
                "subject_name": row.get("subject_name", ""),
                "topic_name": row.get("topic_name", ""),
                "choice_type": row.get("choice_type", ""),
                "id": row.get("id", ""),
                "exp": row.get("exp", ""),
            },
        )
