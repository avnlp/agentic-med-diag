"""PubMedQA dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import MCQSample, QASample
from am_diag.loaders.dataset.base import QADataset
from am_diag.loaders.dataset.prompt import format_mcq_prompt
from am_diag.loaders.dataset.shuffle_utils import shuffle_options


_DATA_DIR = Path(__file__).parent / "data"
_TEST_IDS: frozenset[str] = frozenset(
    json.loads((_DATA_DIR / "pubmedqa_test_ids.json").read_text()).keys(),
)

_DECISION_MAP: dict[str, str] = {"yes": "A", "no": "B", "maybe": "C"}
_OPTIONS: dict[str, str] = {"A": "Yes", "B": "No", "C": "Maybe"}


class PubMedQADataset(QADataset):
    """Loader for the PubMedQA dataset.

    The HF `pqa_labeled` split has 1,000 rows; this loader filters to exactly
    the 500 IDs listed in `data/pubmedqa_test_ids.json` to match the published
    benchmark. Options are always the fixed 3-way set: Yes / No / Maybe.
    """

    dataset_name: ClassVar[str] = "pubmedqa"
    hf_repo: ClassVar[str] = "qiaojin/PubMedQA"
    default_split: ClassVar[str] = "test"

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the PubMedQA dataset.

        Args:
            split: Dataset split (`"test"` or `"train"`).
            limit: Maximum number of samples.

        Returns:
            List of MCQSample instances.
        """
        target_split = split or self.default_split
        if target_split == "train":
            hf_config, hf_split = "pqa_artificial", "train"
        else:
            hf_config, hf_split = "pqa_labeled", "train"

        ds = load_dataset(self.hf_repo, hf_config, split=hf_split, streaming=True)
        if limit is not None:
            ds = ds.take(limit)
        result: list[QASample] = []
        for idx, row in enumerate(ds):
            sample = self._row_to_sample(idx, row, target_split)
            if sample is not None:
                result.append(sample)
        return result

    def _row_to_sample(self, idx: int, row: Any, split: str) -> MCQSample | None:
        question_text = (row.get("question") or "").strip()
        if not question_text:
            return None

        pubid = str(row.get("pubid") or "")
        if split != "train" and pubid not in _TEST_IDS:
            return None

        final_decision = (row.get("final_decision") or "").lower().strip()
        answer_key = _DECISION_MAP.get(final_decision)
        if answer_key is None:
            return None

        context = row.get("context") or {}
        labels: list[Any] = (
            context.get("labels") or [] if isinstance(context, dict) else []
        )
        contexts: list[Any] = (
            context.get("contexts") or [] if isinstance(context, dict) else []
        )
        formatted = "\n".join(
            f"{lbl}. {ctx}" for lbl, ctx in zip(labels, contexts, strict=False)
        )
        question_text = row.get("question") or ""
        question_stem = f"Context:\n{formatted}\n\nQuestion: {question_text}".strip()

        options = dict(_OPTIONS)
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

        meshes: list[Any] = []
        if isinstance(context, dict):
            meshes = context.get("meshes") or []

        return MCQSample(
            sample_id=pubid or str(idx),
            question=question,
            question_stem=question_stem,
            options=options,
            options_original=options_original,
            answer_key=answer_key,
            answer_text=answer_text,
            answer=answer_key,
            dataset=self.dataset_name,
            split=split,
            metadata={"pubid": pubid, "meshes": meshes},
        )
