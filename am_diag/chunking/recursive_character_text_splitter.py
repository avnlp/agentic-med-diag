"""Recursive-character text chunker based on LangChain's Text Splitter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.document import Document


class RecursiveCharacterTextChunker:
    r"""Splits text documents into chunks using LangChain's Text Splitter.

    Produces `Chunk` DataPoints with full provenance back to the
    source `Document`.

    Parameters:
        separators: Ordered list of separator strings to try.  When
            `None` the splitter defaults to
            `["\\n\\n", "\\n", " ", ""]`.
        keep_separator: Whether to keep the separator in the output.
            `True` or `"start"` prepends it to the next chunk;
            `"end"` appends it to the previous chunk.
        is_separator_regex: If `True`, separators are treated as raw
            regex patterns instead of being escaped.
        chunk_size: Maximum character count (per `length_function`)
            of each chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.
        length_function: Callable that returns the length of a string.
            Defaults to `len` (character count).
        add_start_index: When `True`, includes a `"start_index"`
            key in each chunk's `properties` dict.
        strip_whitespace: Strip leading/trailing whitespace from each
            output chunk.
    """

    def __init__(  # noqa: PLR0913
        self,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = True,
        is_separator_regex: bool = False,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize the chunker with splitter parameters.

        See the class docstring for parameter descriptions.
        """
        kwargs: dict = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "length_function": length_function,
            "add_start_index": add_start_index,
            "strip_whitespace": strip_whitespace,
        }
        if separators is not None:
            kwargs["separators"] = separators
        kwargs["keep_separator"] = keep_separator
        kwargs["is_separator_regex"] = is_separator_regex
        self._splitter = RecursiveCharacterTextSplitter(**kwargs)
        self._add_start_index = add_start_index

    async def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split each document into `Chunk` DataPoints with full provenance.

        Args:
            documents: Source documents to chunk.

        Returns:
            Flat list of `Chunk` objects across all input documents,
            ordered by document then by position within the document.
        """
        return await asyncio.to_thread(self._split, documents)

    def _split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            lc_docs = self._splitter.create_documents(
                [doc.text],
                metadatas=[{"doc_id": str(doc.id), "source": doc.source}],
            )
            for idx, lc_doc in enumerate(lc_docs):
                props: dict = {}
                if self._add_start_index:
                    start = lc_doc.metadata.get("start_index")
                    if start is not None:
                        props["start_index"] = start
                chunks.append(
                    Chunk(
                        text=lc_doc.page_content,
                        document_id=doc.id,
                        document_source=doc.source,
                        chunk_index=idx,
                        chunk_size=len(lc_doc.page_content),
                        cut_type="recursive",
                        properties=props,
                        title=doc.title,
                    )
                )
        return chunks
