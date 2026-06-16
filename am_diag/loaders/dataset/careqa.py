"""CareQA MCQ dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import MCQSample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


class CareQADataset(QADataset):
    """Loader for the MCQ split of HPAI-BSC/CareQA (`CareQA_en` config)."""

    dataset_name: ClassVar[str] = "careqa"
    hf_repo: ClassVar[str] = "HPAI-BSC/CareQA"
    hf_config: ClassVar[str | None] = "CareQA_en"
    default_split: ClassVar[str] = "test"

    def _row_to_sample(self, idx: int, row: Any, split: str) -> MCQSample | None:
        question_stem = (row.get("question") or "").strip()
        if not question_stem:
            return None
        options = {
            "A": str(row.get("op1") or ""),
            "B": str(row.get("op2") or ""),
            "C": str(row.get("op3") or ""),
            "D": str(row.get("op4") or ""),
        }
        cop = row.get("cop")
        if cop not in (1, 2, 3, 4):
            return None
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
            sample_id=str(row.get("id", idx)),
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
                "subject": row.get("subject", ""),
                "id": row.get("id", ""),
            },
        )
