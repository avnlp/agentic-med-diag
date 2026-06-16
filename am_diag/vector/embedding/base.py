"""Abstract base class for text embedders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import Self


class Embedder(ABC):
    """Contract for text-embedding clients used in ingestion and query pipelines.

    Concrete subclasses implement `embed` and `close`. The base
    class provides async context manager support so embedders can be used
    as::

        async with SentenceTransformersEmbedder() as embedder:
            vectors = await embedder.embed(["disease X causes symptom Y"])
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: Texts to embed. An empty list yields an empty result.

        Returns:
            One embedding vector per input text, in the same order as
            `texts`. Each vector is a `list[float]`.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release the underlying client/model and any associated resources."""

    async def __aenter__(self) -> Self:
        """Enter async context; return self for use as an async context manager."""
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit async context; close the embedder and release resources."""
        await self.close()
