"""Abstract base class for text rerankers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Self


class RerankResult(BaseModel):
    """A single result from a reranking operation.

    Attributes:
        index: Position of this document in the original ``documents`` list
            passed to ``Reranker.rerank``.
        score: Calibrated relevance score in ``[0, 1]`` (higher is more
            relevant).
        text: The document text payload.
    """

    index: int
    score: float = Field(ge=0.0, le=1.0)
    text: str


class Reranker(ABC):
    """Contract for reranking text documents against a query.

    Concrete subclasses implement `rerank` and `close`. The base
    class provides async context manager support so rerankers can be used
    as::

        async with SentenceTransformersReranker() as reranker:
            results = await reranker.rerank(query, documents)
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Re-rank `documents` by relevance to `query`.

        Args:
            query: The search query or question.
            documents: Text documents to re-rank. An empty list yields an
                empty result.
            top_k: If set, only the top-`k` highest-scoring documents are
                returned. When `None`, all documents are returned ordered
                by descending score.

        Returns:
            Documents ordered by descending relevance score. Each result
            carries the original `index`, a calibrated `score` in `[0, 1]`,
            and the `text` payload.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release the underlying model and any associated resources."""

    async def __aenter__(self) -> Self:
        """Enter async context; return self for use as an async context manager."""
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit async context; close the reranker and release resources."""
        await self.close()
