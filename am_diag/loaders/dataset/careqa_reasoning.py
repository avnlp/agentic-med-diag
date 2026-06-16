"""CareQA open-ended reasoning dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import OpenEndedSample
from am_diag.loaders.dataset.base import QADataset


class CareQAReasoningDataset(QADataset):
    """Loader for the open-ended reasoning split of CareQA dataset."""

    dataset_name: ClassVar[str] = "careqa_reasoning"
    hf_repo: ClassVar[str] = "HPAI-BSC/CareQA"
    hf_config: ClassVar[str | None] = "CareQA_en_open"
    default_split: ClassVar[str] = "test"

    def _row_to_sample(self, idx: int, row: Any, split: str) -> OpenEndedSample | None:
        question = (row.get("question") or "").strip()
        if not question:
            return None
        reference_answer = str(row.get("answer_explanation") or row.get("answer") or "")
        return OpenEndedSample(
            sample_id=str(idx),
            question=question,
            reference_answer=reference_answer,
            dataset=self.dataset_name,
            split=split,
            answer=reference_answer,
            metadata={"subject": row.get("subject", "")},
        )
