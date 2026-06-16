"""Reranker backed by a local `sentence-transformers` CrossEncoder model."""

from __future__ import annotations

import asyncio
import logging
from typing import Any


try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError as _exc:
    raise ImportError(
        "sentence-transformers requires: pip install 'am_diag[sentence-transformers]'",
    ) from _exc

from am_diag.vector.reranking.base import Reranker, RerankResult


logger = logging.getLogger(__name__)


class SentenceTransformersReranker(Reranker):
    r"""Rerank documents using a local ``sentence_transformers.CrossEncoder`` model.

    The model is loaded lazily on first use and cached for subsequent calls.
    Because ``CrossEncoder.predict`` is synchronous and GPU-bound, calls are
    executed in a worker thread via ``asyncio.to_thread`` so the event loop
    is not blocked.

    Parameters:
        model_name: Name or path of a CrossEncoder model, e.g.
            ``"zeroentropy/zerank-2-reranker"``.
        device: Device for model inference, e.g. ``"cpu"`` or ``"cuda"``.
        batch_size: Maximum query-document pairs per internal ``predict`` call.
        max_length: Maximum token length for input pairs (``CrossEncoder``
            truncates beyond this).
        top_k: Default number of top results to return when ``top_k`` is not
            explicitly passed to ``rerank()``.
        trust_remote_code: Allow the HF model to load custom remote code.
        torch_dtype: Optional torch dtype string, e.g. ``"bfloat16"``.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = "zeroentropy/zerank-2-reranker",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 32768,
        top_k: int = 10,
        trust_remote_code: bool = False,
        torch_dtype: str | None = None,
    ) -> None:
        """Store configuration; the model is loaded lazily on first ``rerank`` call."""
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._top_k = top_k
        self._trust_remote_code = trust_remote_code
        self._torch_dtype = torch_dtype

        self._model: CrossEncoder | None = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> CrossEncoder:
        """Lazy-initialise the underlying model with double-checked locking."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    kwargs: dict[str, Any] = {
                        "device": self._device,
                        "max_length": self._max_length,
                    }
                    if self._trust_remote_code:
                        kwargs["trust_remote_code"] = True
                    if self._torch_dtype is not None:
                        kwargs["torch_dtype"] = self._torch_dtype
                    self._model = await asyncio.to_thread(
                        lambda: CrossEncoder(self._model_name, **kwargs),
                    )
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
        instruction: str | None = None,
    ) -> list[RerankResult]:
        """Re-rank ``documents`` by relevance to ``query``.

        Args:
            query: The search query or question.
            documents: Text documents to re-rank. An empty list short-circuits
                without touching the model.
            top_k: If set, only the top-``k`` highest-scoring documents are
                returned. When ``None``, the constructor's ``top_k`` default
                is used.
            instruction: Optional natural-language instruction appended as an
                ``<instruction>`` tag to influence relevance assessment
                (supported by zerank-2 and similar instruction-following
                cross-encoders).

        Returns:
            Documents ordered by descending relevance score.
        """
        if not documents:
            return []

        if not query.strip():
            return [
                RerankResult(index=i, score=0.0, text=d)
                for i, d in enumerate(documents)
            ]

        model = await self._get_model()
        return await asyncio.to_thread(
            self._rerank,
            model,
            query,
            documents,
            top_k,
            instruction,
        )

    def _rerank(
        self,
        model: CrossEncoder,
        query: str,
        documents: list[str],
        top_k: int | None,
        instruction: str | None,
    ) -> list[RerankResult]:
        """Run cross-encoder scoring in a worker thread."""
        effective_query = query
        if instruction:
            effective_query = f"{query}\n<instruction>{instruction}</instruction>"

        pairs = [(effective_query, doc) for doc in documents]

        raw_scores = model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        scores: list[float] = (torch.tensor(raw_scores) / 5.0).sigmoid().tolist()

        results = [
            RerankResult(index=i, score=s, text=documents[i])
            for i, s in enumerate(scores)
        ]

        results.sort(key=lambda r: r.score, reverse=True)

        k = top_k if top_k is not None else self._top_k
        return results[:k]

    async def close(self) -> None:
        """Release the underlying model reference."""
        self._model = None
