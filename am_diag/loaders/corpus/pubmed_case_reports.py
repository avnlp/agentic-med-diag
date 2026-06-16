"""Loader for the PubMed Case Reports corpus."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.base import CorpusLoader


class PubmedCaseReportsCorpusLoader(CorpusLoader):
    """Load the PubMed Case Reports corpus."""

    corpus_name: ClassVar[str] = "pubmed_case_reports"
    hf_repo: ClassVar[str] = "zou-lab/MedCaseReasoning"
    hf_split: ClassVar[str] = "train"

    def _row_to_document(self, row: dict[str, Any]) -> Document | None:
        """Map one MedCaseReasoning row to a Document.

        Args:
            row: Raw row with `pmcid`, `title`, `journal`,
                `publication_date`, `text` keys.

        Returns:
            A Document, or `None` if the article text is empty.
        """
        text = row.get("text")
        if not text:
            return None

        pmcid = row["pmcid"]
        return Document(
            text=text,
            source="pubmed_case_reports",
            external_id=pmcid,
            title=row.get("title"),
            properties={
                "pmcid": pmcid,
                "journal": row.get("journal", ""),
                "publication_date": row.get("publication_date", ""),
                "section_type": "full_text",
                "license": "CC-BY-4.0",
            },
        )
