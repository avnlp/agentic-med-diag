"""Load the PubMed Abstracts corpus."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.base import CorpusLoader


class PubMedCorpusLoader(CorpusLoader):
    """Load the PubMed Abstracts corpus.

    Loads the `MedRAG/pubmed` corpus from HuggingFace.
    """

    corpus_name: ClassVar[str] = "pubmed"
    hf_repo: ClassVar[str] = "MedRAG/pubmed"
    hf_split: ClassVar[str] = "train"

    def _row_to_document(self, row: dict[str, Any]) -> Document | None:
        """Map one MedRAG/pubmed row to a Document.

        Args:
            row: Raw row with `id`, `title`, `content` keys.

        Returns:
            A Document, or `None` if the content is empty.
        """
        text = row.get("content")
        if not text:
            return None

        doc_id = row["id"]
        # id format: "pubmed-<pubmed_id>-<chunk_index>" (hyphen-separated).
        stripped = doc_id.removeprefix("pubmed-")
        pubmed_id, _, index_str = stripped.rpartition("-")
        if not pubmed_id:
            pubmed_id = stripped
            index_str = ""
        try:
            chunk_index = int(index_str)
        except ValueError:
            chunk_index = 0

        return Document(
            text=text,
            source="pubmed",
            external_id=doc_id,
            title=row.get("title"),
            properties={"pubmed_id": pubmed_id, "chunk_index": chunk_index},
        )
