"""Abstract base class and shared streaming pattern for corpus loaders."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, ClassVar

from datasets import load_dataset

from am_diag.common.data_models import Document


class CorpusLoader(ABC):
    """Abstract base for async, streaming medical corpus loaders.

    Every corpus loader subclasses `CorpusLoader` and implements
    `_row_to_document` (and, for loaders whose one source row expands into many
    documents, overrides `astream`). The base class supplies the standard async
    streaming loop that wraps HuggingFace's synchronous iterator in
    `asyncio.to_thread` so it never blocks a agent async node's event loop.

    Attributes:
        corpus_name: Short corpus identifier (matches `Document.source`).
        hf_repo: HuggingFace dataset repo id to stream from.
        hf_split: Dataset split to load (defaults to `"train"`).
    """

    corpus_name: ClassVar[str]
    hf_repo: ClassVar[str]
    hf_split: ClassVar[str] = "train"

    @abstractmethod
    def _row_to_document(self, row: dict[str, Any]) -> Document | None:
        """Convert a single raw dataset row into a Document.

        Args:
            row: One row from the HuggingFace dataset.

        Returns:
            A populated Document, or `None` if the row is empty
            or otherwise invalid and should be skipped.
        """

    async def astream(
        self,
        batch_size: int = 100,
        offset: int = 0,
        limit: int | None = None,
    ) -> AsyncIterator[list[Document]]:
        """Stream the corpus as successive batches of documents.

        The HuggingFace streaming iterator is synchronous; each batch is
        collected inside `asyncio.to_thread` so the event loop stays free for
        concurrent async ingestion work.

        Args:
            batch_size: Maximum number of documents per yielded batch.
            offset: Number of source rows to skip before streaming. Note that
                `Dataset.skip` is O(offset) for streaming datasets, so large
                offsets are slow. For production resumption, prefer recording the
                last processed `doc_id` in checkpoint state and filtering by
                it instead of relying on `offset`.
            limit: Maximum number of source rows to read (`None` for all).

        Yields:
            Lists of Document of length up to `batch_size`.
            Invalid rows (where `_row_to_document` returns `None`) are dropped,
            so a batch may be shorter than `batch_size`.
        """

        def _collect_batch(
            ds_iter: Iterator[dict[str, Any]],
            n: int,
        ) -> list[Document]:
            # Pull up to `n` rows synchronously and convert them. Runs in a
            # worker thread via asyncio.to_thread so it does not block the loop.
            batch: list[Document] = []
            try:
                for _ in range(n):
                    row = next(ds_iter)
                    doc = self._row_to_document(row)
                    if doc is not None:
                        batch.append(doc)
            except StopIteration:
                pass
            return batch

        ds = load_dataset(self.hf_repo, split=self.hf_split, streaming=True)
        if offset > 0:
            ds = ds.skip(offset)
        if limit is not None:
            ds = ds.take(limit)

        ds_iter = iter(ds)
        while True:
            batch = await asyncio.to_thread(_collect_batch, ds_iter, batch_size)
            if not batch:
                break
            yield batch

    async def load_sample(self, n: int, offset: int = 0) -> list[Document]:
        """Load a flat list of up to `n` documents. For dev/testing only.

        Args:
            n: Number of documents to load.
            offset: Number of source rows to skip first.

        Returns:
            A list of up to `n` Document objects.
        """
        result: list[Document] = []
        async for batch in self.astream(batch_size=n, offset=offset, limit=n):
            result.extend(batch)
            break
        return result

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return f"{type(self).__name__}(corpus={self.corpus_name!r})"
