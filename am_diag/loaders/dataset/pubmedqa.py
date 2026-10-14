"""PubMedQA dataset loader."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


_DATA_DIR = Path(__file__).parent / "data"
_TEST_IDS: frozenset[str] = frozenset(
    json.loads((_DATA_DIR / "pubmedqa_test_ids.json").read_text()).keys(),
)

_DECISION_MAP: dict[str, str] = {"yes": "A", "no": "B", "maybe": "C"}
_OPTIONS: dict[str, str] = {"A": "Yes", "B": "No", "C": "Maybe"}


class PubMedQADataset(MCQDataset):
    """Loader for the PubMedQA dataset.

    The HF `pqa_labeled` split has 1,000 rows; this loader filters to exactly
    the 500 IDs listed in `data/pubmedqa_test_ids.json` to match the published
    benchmark. Options are always the fixed 3-way set: Yes / No / Maybe.
    """

    dataset_name: ClassVar[str] = "pubmedqa"
    hf_repo: ClassVar[str] = "qiaojin/PubMedQA"
    default_split: ClassVar[str] = "test"

    def _rows(self, split: str) -> Iterable[Any]:
        hf_config = "pqa_artificial" if split == "train" else "pqa_labeled"
        ds = load_dataset(self.hf_repo, hf_config, split="train", streaming=True)
        for row in ds:
            if split != "train" and str(row.get("pubid") or "") not in _TEST_IDS:
                continue
            yield row

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:
        question_text = (record.get("question") or "").strip()
        if not question_text:
            return None

        final_decision = (record.get("final_decision") or "").lower().strip()
        key = _DECISION_MAP.get(final_decision)
        if key is None:
            return None

        context = record.get("context") or {}
        labels: list[Any] = (
            context.get("labels") or [] if isinstance(context, dict) else []
        )
        contexts: list[Any] = (
            context.get("contexts") or [] if isinstance(context, dict) else []
        )
        formatted = "\n".join(
            f"{lbl}. {ctx}" for lbl, ctx in zip(labels, contexts, strict=False)
        )
        stem = (
            f"Context:\n{formatted}\n\nQuestion: {record.get('question') or ''}".strip()
        )

        meshes: list[Any] = (
            context.get("meshes") or [] if isinstance(context, dict) else []
        )
        pubid = str(record.get("pubid") or "")
        return ExtractedMCQ(
            question_stem=stem,
            options=dict(_OPTIONS),
            answer_keys=[key],
            metadata={"pubid": pubid, "meshes": meshes},
            sample_id=pubid or str(idx),
        )
