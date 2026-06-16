"""RAR-Medicine dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import RARMedSample
from am_diag.loaders.dataset.base import QADataset


class RARMedDataset(QADataset):
    """Loader for the RAR-Medicine dataset."""

    dataset_name: ClassVar[str] = "rar_med"
    hf_repo: ClassVar[str] = "anisha2102/RaR-Medicine"
    default_split: ClassVar[str] = "train"

    def _row_to_sample(self, idx: int, row: Any, split: str) -> RARMedSample | None:
        question = (row.get("question") or "").strip()
        if not question:
            return None
        reference_answer = str(row.get("reference_answer") or "")
        rubrics: list[dict[str, Any]] = list(row.get("rubric") or [])
        excluded = {"question", "reference_answer", "rubric"}
        metadata: dict[str, Any] = {}
        for key in row:
            if key not in excluded:
                metadata[str(key)] = row[key]
        return RARMedSample(
            sample_id=str(idx),
            question=question,
            reference_answer=reference_answer,
            rubrics=rubrics,
            answer=reference_answer,
            metadata=metadata,
        )
