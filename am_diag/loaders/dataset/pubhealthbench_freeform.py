"""PubHealthBench freeform dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import OpenEndedSample
from am_diag.loaders.dataset.base import QADataset


class PubHealthBenchFreeformDataset(QADataset):
    """Loader for the PubHealthBench freeform dataset."""

    dataset_name: ClassVar[str] = "pubhealthbench_freeform"
    hf_repo: ClassVar[str] = "Joshua-Harris/PubHealthBench"
    default_split: ClassVar[str] = "test"

    def _row_to_sample(self, idx: int, row: Any, split: str) -> OpenEndedSample | None:
        question = (row.get("question") or "").strip()
        if not question:
            return None
        # Derive reference answer from the correct option text when possible.
        options: list[Any] = row.get("options") or []
        answer_index = row.get("answer_index")
        if options and answer_index is not None:
            try:
                reference_answer = str(options[int(answer_index)])
            except (IndexError, TypeError, ValueError):
                reference_answer = str(row.get("reference_answer") or "")
        else:
            reference_answer = str(row.get("reference_answer") or "")
        return OpenEndedSample(
            sample_id=str(row.get("question_id", idx)),
            question=question,
            reference_answer=reference_answer,
            dataset=self.dataset_name,
            split=split,
            answer=reference_answer,
            metadata={
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "retrieved_context_for_judge": row.get("retrieved_context_for_judge")
                or "",
            },
        )
