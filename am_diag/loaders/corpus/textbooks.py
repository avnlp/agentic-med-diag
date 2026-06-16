"""Load the USMLE Medical Textbooks."""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.common.data_models import Document
from am_diag.loaders.corpus.base import CorpusLoader


class TextbooksCorpusLoader(CorpusLoader):
    """Load the USMLE Medical Textbooks.

    Load the `MedRAG/textbooks` corpus from HuggingFace.
    """

    corpus_name: ClassVar[str] = "textbooks"
    hf_repo: ClassVar[str] = "MedRAG/textbooks"
    hf_split: ClassVar[str] = "train"

    def _row_to_document(self, row: dict[str, Any]) -> Document | None:
        """Map one MedRAG/textbooks row to a Document.

        Args:
            row: Raw row with `id`, `title`, `content` keys.
                - `id`: "textbooks-<Book_Name>_<chunk_index>"
                - `title`: Book title (optional)
                - `content`: Chunk text

        Returns:
            A Document, or `None` if the content is empty.
        """
        text = row.get("content")
        if not text:
            return None

        doc_id = row["id"]
        # The id is "textbooks-<Book_Name>_<chunk_index>".
        body, _, index_str = doc_id.rpartition("_")
        book_name = body.removeprefix("textbooks-")
        try:
            chunk_index = int(index_str)
        except ValueError:
            chunk_index = 0

        return Document(
            text=text,
            source="textbooks",
            external_id=doc_id,
            title=row.get("title", book_name),
            properties={"book_name": book_name, "chunk_index": chunk_index},
        )
