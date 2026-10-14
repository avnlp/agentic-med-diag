"""CareQA MCQ dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


class CareQADataset(MCQDataset):
    """Loader for the MCQ split of HPAI-BSC/CareQA (`CareQA_en` config)."""

    dataset_name: ClassVar[str] = "careqa"
    hf_repo: ClassVar[str] = "HPAI-BSC/CareQA"
    hf_config: ClassVar[str | None] = "CareQA_en"
    default_split: ClassVar[str] = "test"

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        stem = (record.get("question") or "").strip()
        if not stem:
            return None
        options = {
            "A": str(record.get("op1") or ""),
            "B": str(record.get("op2") or ""),
            "C": str(record.get("op3") or ""),
            "D": str(record.get("op4") or ""),
        }
        cop = record.get("cop")
        if cop not in (1, 2, 3, 4):
            return None
        key = ["A", "B", "C", "D"][int(cop) - 1]
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={
                "subject": record.get("subject", ""),
                "id": record.get("id", ""),
            },
            sample_id=str(record.get("id", idx)),
        )
