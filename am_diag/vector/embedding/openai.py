"""Embedder backed by the OpenAI embeddings API."""

from __future__ import annotations

import asyncio
import logging


try:
    from openai import AsyncOpenAI
except ImportError as _exc:
    raise ImportError(
        "OpenAI support requires: pip install 'am_diag[openai]'",
    ) from _exc

from am_diag.vector.embedding.base import Embedder


logger = logging.getLogger(__name__)


class OpenAIEmbedder(Embedder):
    """Embed text using the OpenAI embeddings API via ``openai.AsyncOpenAI``.

    The client is created lazily on first use and cached for subsequent
    calls. Texts are sent in batches of ``batch_size`` to respect
    request-size limits.

    Parameters:
        model_name: OpenAI embedding model name, e.g.
            ``"text-embedding-3-small"``.
        batch_size: Maximum texts per API request.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 32,
    ) -> None:
        """Store configuration; the client is created lazily on first ``embed`` call."""
        self._model_name = model_name
        self._batch_size = batch_size

        self._client: AsyncOpenAI | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialise the OpenAI client with double-checked locking."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = AsyncOpenAI()
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the OpenAI embeddings API.

        Args:
            texts: Texts to embed. An empty list short-circuits without
                making any API calls.

        Returns:
            One embedding vector per input text, in the same order as
            ``texts``.

        Raises:
            openai.OpenAIError: On API or connectivity failure.
        """
        if not texts:
            return []
        client = await self._get_client()
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            response = await client.embeddings.create(
                model=self._model_name,
                input=chunk,
            )
            embeddings.extend(item.embedding for item in response.data)
            logger.debug("Embedded %d text(s) via OpenAI", len(chunk))
        return embeddings

    async def close(self) -> None:
        """Close the underlying OpenAI HTTP client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
