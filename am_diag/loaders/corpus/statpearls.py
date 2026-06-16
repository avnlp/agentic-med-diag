"""Load the StatPearls Clinical Guidelines corpus."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.base import CorpusLoader


class StatPearlsCorpusLoader(CorpusLoader):
    """Load the StatPearls Clinical Guidelines corpus.

    Each HF row is a single chunk from a StatPearls article.
    """

    corpus_name: ClassVar[str] = "statpearls"
    hf_repo: ClassVar[str] = "awinml/statpearls"
    hf_split: ClassVar[str] = "train"

    def _row_to_document(self, row: dict[str, Any]) -> Document | None:
        """Map one StatPearls row to a Document.

        Args:
            row: Raw row with `id`, `title`, `content` keys.

        Returns:
            A Document, or `None` if the content is empty.
        """
        text = row.get("content")
        if not text:
            return None

        doc_id = row["id"]
        # The `id` encodes the article id and chunk index (e.g. "article-100024_2")
        # id format: "article-<article_id>_<chunk_index>".
        stripped = doc_id.removeprefix("article-")
        article_id, _, index_str = stripped.rpartition("_")
        if not article_id:
            # No "_N" suffix present; treat the whole remainder as the id.
            article_id = stripped
            index_str = ""
        try:
            chunk_index = int(index_str)
        except ValueError:
            chunk_index = 0

        # Title encodes the article name and section
        # (e.g. "Chronic Occlusion -- Introduction")
        title = row.get("title")
        if title and " -- " in title:
            section_type = title.rsplit(" -- ", 1)[-1]
        else:
            section_type = "Unknown"

        return Document(
            text=text,
            source="statpearls",
            external_id=doc_id,
            title=title,
            properties={
                "article_id": article_id,
                "section_type": section_type,
                "chunk_index": chunk_index,
            },
        )
