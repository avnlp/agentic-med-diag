"""NEJM Diagnostic Reasoning open-ended dataset loader."""

from __future__ import annotations

from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import OpenEndedSample, QASample
from am_diag.loaders.dataset.base import QADataset


class NEJMDiagnosticDataset(QADataset):
    """Loader for the NEJM Diagnostic Reasoning open-ended dataset."""

    dataset_name: ClassVar[str] = "nejm_diagnostic"
    hf_repo: ClassVar[str] = "katielink/nejm-medqa-diagnostic-reasoning-dataset"
    default_split: ClassVar[str] = "nejm_test"

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load NEJM diagnostic reasoning questions.

        Args:
            split: HF config name (`nejm_test`, `medqa_test`,
                `medqa_train`). Defaults to `nejm_test`.
            limit: Maximum number of samples.

        Returns:
            List of OpenEndedSample instances.
        """
        hf_config = split or self.default_split
        if hf_config not in ("nejm_test", "medqa_test", "medqa_train"):
            msg = (
                f"Unknown config '{hf_config}'. "
                "Expected: nejm_test, medqa_test, medqa_train"
            )
            raise ValueError(msg)
        ds = load_dataset(self.hf_repo, hf_config, split="train", streaming=True)
        if limit is not None:
            ds = ds.take(limit)
        result: list[QASample] = []
        for idx, row in enumerate(ds):
            sample = self._row_to_sample(idx, row, hf_config)  # type: ignore[arg-type]
            if sample is not None:
                result.append(sample)
        return result

    def _row_to_sample(self, idx: int, row: Any, split: str) -> OpenEndedSample | None:
        question = (
            row.get("case_presentation")
            or row.get("Title")
            or row.get("question")
            or ""
        ).strip()
        if not question:
            return None
        reference_answer = str(
            row.get("diagnosis") or row.get("Answer") or row.get("answer") or "",
        )
        reasoning_raw = row.get("discussion") or row.get("Lab Data")
        reasoning_chain: str | None = str(reasoning_raw) if reasoning_raw else None
        return OpenEndedSample(
            sample_id=str(idx),
            question=question,
            reference_answer=reference_answer,
            reasoning_chain=reasoning_chain,
            dataset=self.dataset_name,
            split=split,
            answer=reference_answer,
            metadata={
                "specialty": row.get("specialty", ""),
                "source": row.get("source", ""),
            },
        )
