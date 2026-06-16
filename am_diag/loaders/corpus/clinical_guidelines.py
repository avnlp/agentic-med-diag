"""Loader for the Meditron Clinical Practice Guidelines corpus."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.base import CorpusLoader


class ClinicalGuidelinesCorpusLoader(CorpusLoader):
    """Load the Meditron Clinical Practice Guidelines corpus.

    Loads the `epfl-llm/guidelines` dataset from HuggingFace.
    """

    corpus_name: ClassVar[str] = "clinical_guidelines"
    hf_repo: ClassVar[str] = "epfl-llm/guidelines"
    hf_split: ClassVar[str] = "train"

    def _row_to_document(self, row: dict[str, Any]) -> Document | None:
        """Map one guidelines row to a Document.

        Args:
            row: Raw row with `id`, `source`, `title`, `clean_text`,
                `url`, `overview` keys.

        Returns:
            A Document, or `None` if the text is empty.
        """
        text = row.get("clean_text")
        if not text:
            return None

        issuing_org = row.get("source", "")
        return Document(
            text=text,
            source="clinical_guidelines",
            external_id=row["id"],
            title=row.get("title"),
            properties={
                "guideline_id": row["id"],
                "issuing_org": issuing_org,
                "url": row.get("url"),
                "overview": (row.get("overview") or "")[:500],
                "section_type": "body",
            },
        )
