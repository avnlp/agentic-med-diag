"""MedQA dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


class MedQADataset(MCQDataset):
    """Loader for the MedQA multiple-choice question dataset."""

    dataset_name: ClassVar[str] = "medqa"
    hf_repo: ClassVar[str] = "GBaker/MedQA-USMLE-4-options"
    default_split: ClassVar[str] = "test"

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        stem = (record.get("question") or "").strip()
        options: dict[str, str] = record.get("options") or {}
        key = str(record.get("answer_idx") or "").strip().upper()
        if not stem or not options or key not in options:
            return None
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={"meta_info": record.get("meta_info", "")},
        )
