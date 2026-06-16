"""HealthBench dataset loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import (
    ConsensusCriterion,
    ConversationTurn,
    QASample,
    RubricCriterion,
    RubricSample,
)
from am_diag.loaders.dataset.base import QADataset


_DATA_DIR = Path(__file__).parent / "data"
_CONSENSUS_LOOKUP: dict[str, dict[str, str]] = json.loads(
    (_DATA_DIR / "hb_consensus_criteria.json").read_text(),
)

_VARIANT_CONFIG: dict[str, tuple[str, str]] = {
    "all": ("neuralleap/healthbench-regular", "test"),
    "hard": ("neuralleap/healthbench-hard", "train"),
    "consensus": ("neuralleap/healthbench-consensus", "train"),
}


class HealthBenchDataset(QADataset):
    """Loader for the HealthBench rubric-scored benchmark.

    Rows are multi-turn conversations scored against a list of rubric criteria.
    Each criterion has a `points` value (may be negative) and tags that encode
    `axis`, `level`, and `cluster`. For cluster-level criteria, a
    `ConsensusCriterion` is looked up from `hb_consensus_criteria.json`.
    """

    dataset_name: ClassVar[str] = "healthbench"
    hf_repo: ClassVar[str] = "neuralleap/healthbench-regular"
    default_split: ClassVar[str] = "test"

    def __init__(self, variant: str = "all") -> None:
        """Initialise the HealthBench dataset loader.

        Args:
            variant: One of `"all"`, `"hard"`, or `"consensus"`.

        Raises:
            ValueError: If `variant` is not recognised.
        """
        if variant not in _VARIANT_CONFIG:
            raise ValueError(
                f"Unknown variant {variant!r}; choose from {sorted(_VARIANT_CONFIG)}",
            )
        self._variant = variant
        self._hf_repo, self._hf_split = _VARIANT_CONFIG[variant]

    def load(
        self,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[QASample]:
        """Load the HealthBench rubric-scored dataset.

        Args:
            split: Ignored for HealthBench (variant determines split).
            limit: Maximum number of samples.

        Returns:
            List of RubricSample instances.
        """
        ds = load_dataset(self._hf_repo, split=self._hf_split, streaming=True)
        if limit is not None:
            ds = ds.take(limit)
        result: list[QASample] = []
        for idx, row in enumerate(ds):
            sample = self._row_to_sample(idx, row, self._hf_split)
            if sample is not None:
                result.append(sample)
        return result

    def _row_to_sample(
        self,
        idx: int,
        row: Any,
        split: str = "",
    ) -> RubricSample | None:
        raw_rubrics: list[Any] = row.get("rubrics") or []
        rubrics: list[RubricCriterion] = []
        for r in raw_rubrics:
            tags: list[str] = [str(t) for t in (r.get("tags") or [])]
            tag_dict: dict[str, str] = {}
            for t in tags:
                if ":" in t:
                    k, _, v = t.partition(":")
                    tag_dict[k] = v
            rubrics.append(
                RubricCriterion(
                    criterion=str(r.get("criterion", "")),
                    points=int(r.get("points", 0)),
                    tags=tags,
                    axis=tag_dict.get("axis"),
                    level=tag_dict.get("level"),
                    cluster=tag_dict.get("cluster"),
                ),
            )

        example_tags: list[str] = [str(t) for t in (row.get("example_tags") or [])]
        theme: str | None = None
        for t in example_tags:
            if t.startswith("theme:"):
                theme = t[len("theme:") :]
                break

        criteria = [r.criterion for r in rubrics]
        points_list = [r.points for r in rubrics]
        axes = [r.axis or "" for r in rubrics]
        criterion_ids = [
            hashlib.blake2b(c.encode(), digest_size=8).hexdigest() for c in criteria
        ]
        consensus_criteria: list[ConsensusCriterion | None] = []
        for r in rubrics:
            if r.cluster and r.cluster in _CONSENSUS_LOOKUP:
                cc = _CONSENSUS_LOOKUP[r.cluster]
                consensus_criteria.append(
                    ConsensusCriterion(
                        theme=cc["theme"],
                        behavior_category=cc["behavior_category"],
                        criterion=cc["criterion"],
                    ),
                )
            else:
                consensus_criteria.append(None)

        raw_turns: list[Any] = row.get("prompt") or []
        conversation = [
            ConversationTurn(
                role=str(t.get("role", "")),
                content=str(t.get("content", "")),
            )
            for t in raw_turns
        ]
        question = "\n\n".join(f"{t.role}: {t.content}" for t in conversation)

        return RubricSample(
            sample_id=str(row.get("prompt_id", idx)),
            conversation=conversation,
            rubrics=rubrics,
            theme=theme,
            variant=self._variant,
            criteria=criteria,
            points_list=points_list,
            axes=axes,
            criterion_ids=criterion_ids,
            consensus_criteria=consensus_criteria,
            question=question,
            answer="",
            metadata={},
        )
