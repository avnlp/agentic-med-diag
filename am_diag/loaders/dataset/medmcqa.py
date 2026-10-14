"""MedMCQA dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


class MedMCQADataset(MCQDataset):
    """Loader for the MedMCQA multiple-choice question dataset."""

    dataset_name: ClassVar[str] = "medmcqa"
    hf_repo: ClassVar[str] = "lighteval/med_mcqa"
    default_split: ClassVar[str] = "validation"

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        stem = (record.get("question") or "").strip()
        if not stem:
            return None
        opa = (record.get("opa") or "").strip()
        opb = (record.get("opb") or "").strip()
        opc = (record.get("opc") or "").strip()
        opd = (record.get("opd") or "").strip()
        if not (opa or opb or opc or opd):
            return None
        cop = record.get("cop")
        if cop not in (1, 2, 3, 4):
            return None
        options = {"A": opa, "B": opb, "C": opc, "D": opd}
        key = ["A", "B", "C", "D"][int(cop) - 1]
        return ExtractedMCQ(
            question_stem=stem,
            options=options,
            answer_keys=[key],
            metadata={
                "subject_name": record.get("subject_name", ""),
                "topic_name": record.get("topic_name", ""),
                "choice_type": record.get("choice_type", ""),
                "id": record.get("id", ""),
                "exp": record.get("exp", ""),
            },
            sample_id=str(record.get("id") or idx),
        )
