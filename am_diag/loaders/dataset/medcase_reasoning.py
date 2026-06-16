"""MedCaseReasoning open-ended diagnosis dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import OpenEndedSample
from am_diag.loaders.dataset.base import QADataset


class MedCaseReasoningDataset(QADataset):
    """Loader for the Med Case Reasoning diagnostic reasoning dataset.

    Each sample is a clinical case report paired with a final diagnosis and a
    step-by-step diagnostic reasoning chain.
    """

    dataset_name: ClassVar[str] = "medcase_reasoning"
    hf_repo: ClassVar[str] = "zou-lab/MedCaseReasoning"
    default_split: ClassVar[str] = "val"

    def _row_to_sample(self, idx: int, row: Any, split: str) -> OpenEndedSample | None:
        question = (row.get("case_prompt") or "").strip()
        if not question:
            return None
        reference_answer = str(row.get("final_diagnosis") or "")
        reasoning_raw = row.get("diagnostic_reasoning")
        reasoning_chain: str | None = str(reasoning_raw) if reasoning_raw else None
        return OpenEndedSample(
            sample_id=str(row.get("pmcid", idx)),
            question=question,
            reference_answer=reference_answer,
            reasoning_chain=reasoning_chain,
            dataset=self.dataset_name,
            split=split,
            answer=reference_answer,
            metadata={
                "pmcid": row.get("pmcid", ""),
                "title": row.get("title", ""),
                "journal": row.get("journal", ""),
                "article_link": row.get("article_link", ""),
                "publication_date": row.get("publication_date", ""),
            },
        )
