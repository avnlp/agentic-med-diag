"""Embedder backed by a local `sentence-transformers` model."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal


try:
    from sentence_transformers import SentenceTransformer
except ImportError as _exc:
    raise ImportError(
        "sentence-transformers requires: pip install 'am_diag[sentence-transformers]'",
    ) from _exc

from am_diag.vector.embedding.base import Embedder


logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder(Embedder):
    r"""Embed text using a local ``sentence-transformers`` model.

    The model is loaded lazily on first use and cached for subsequent calls.
    Because ``SentenceTransformer.encode`` is synchronous and CPU/GPU bound,
    calls are executed in a worker thread via ``asyncio.to_thread`` so the
    event loop is not blocked.

    Parameters:
        model_name: Name or path of a SentenceTransformers model, e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``.
        device: Device for model inference, e.g. ``"cpu"`` or ``"cuda"``.
        batch_size: Maximum texts per internal ``encode`` call.
        normalize: L2-normalize output embeddings so cosine similarity equals
            dot product.
        precision: Output precision for the ``encode`` call. Keep ``"float32"
            for vectors persisted to Qdrant / Weaviate; configure quantization
            on the store instead.
        dimensions: Optional truncation dimension (Matryoshka-style). Passed
            as ``truncate_dim`` to the underlying ``encode``.
        input_type: ``"document"`` calls ``model.encode_document``,
            ``"query"`` calls ``model.encode_query``, and ``"generic"`` calls
            ``model.encode``.
        trust_remote_code: Allow the HF model to load custom remote code.
        torch_dtype: Optional torch dtype string passed as ``model_kwargs``,
            e.g. ``"bfloat16"``.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        dimensions: int | None = None,
        input_type: Literal["generic", "query", "document"] = "generic",
        trust_remote_code: bool = False,
        torch_dtype: str | None = None,
    ) -> None:
        """Store configuration; the model is loaded lazily on first ``embed`` call."""
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._precision = precision
        self._dimensions = dimensions
        self._input_type = input_type
        self._trust_remote_code = trust_remote_code
        self._torch_dtype = torch_dtype

        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> SentenceTransformer:
        """Lazy-initialise the underlying model with double-checked locking."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    kwargs: dict[str, Any] = {"device": self._device}
                    if self._trust_remote_code:
                        kwargs["trust_remote_code"] = True
                    if self._torch_dtype is not None:
                        kwargs["model_kwargs"] = {
                            "torch_dtype": self._torch_dtype,
                        }
                    self._model = await asyncio.to_thread(
                        lambda: SentenceTransformer(self._model_name, **kwargs),
                    )
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed. An empty list short-circuits without
                touching the model.

        Returns:
            One embedding vector per input text, in the same order as
            ``texts``.
        """
        if not texts:
            return []
        model = await self._get_model()
        embeddings = await asyncio.to_thread(
            self._encode,
            model,
            texts,
        )
        return [vector.tolist() for vector in embeddings]

    def _encode(self, model: SentenceTransformer, texts: list[str]) -> Any:
        """Synchronous encode dispatch (runs in a worker thread).

        Override this method in a subclass to inject custom logic such as
        projection or quantization.
        """
        kwargs: dict[str, Any] = {
            "batch_size": self._batch_size,
            "normalize_embeddings": self._normalize,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self._precision != "float32":
            kwargs["precision"] = self._precision
        if self._dimensions is not None:
            kwargs["truncate_dim"] = self._dimensions

        encode = model.encode
        if self._input_type == "document":
            encode = model.encode_document
        elif self._input_type == "query":
            encode = model.encode_query

        return encode(texts, **kwargs)

    async def close(self) -> None:
        """Release the underlying model reference."""
        self._model = None
