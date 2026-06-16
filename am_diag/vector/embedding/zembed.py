"""Embedder configured for ZeroEntropy's local zembed-1 model."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from am_diag.vector.embedding.sentence_transformers import SentenceTransformersEmbedder


ZEMBED_MODEL_NAME = "zeroentropy/zembed-1"
ZEMBED_SUPPORTED_DIMS = frozenset({2560, 1280, 640, 320, 160, 80, 40})
_ZEMBED_FULL_DIM = 2560


class ZembedEmbedder(SentenceTransformersEmbedder):
    """SentenceTransformers embedder for the open-weight zembed-1 model.

    zembed-1's lower dimensions are learned projections, not plain
    Matryoshka-style truncation. This wrapper encodes full 2560-dimensional
    vectors first, then applies ZeroEntropy's published projection matrices
    from ``projections.safetensors`` when a lower supported dimension is
    requested.

    Parameters:
        dimensions: Target embedding dimension. Must be one of
            ``{2560, 1280, 640, 320, 160, 80, 40}``.
        precision: Output precision. Keep ``"float32"`` for Qdrant / Weaviate;
            configure quantization on the store instead.
        model_name: zembed model name or path.
        device: Device for model inference.
        batch_size: Maximum texts per internal ``encode`` call.
        normalize: L2-normalize output embeddings.
        trust_remote_code: Allow the HF model to load custom remote code.
        torch_dtype: Optional torch dtype string, e.g. ``"bfloat16"``.
    """

    def __init__(  # noqa: PLR0913
        self,
        dimensions: int = 2560,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        model_name: str = ZEMBED_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
    ) -> None:
        """Store configuration; the model is loaded lazily on first ``embed`` call."""
        if dimensions not in ZEMBED_SUPPORTED_DIMS:
            supported = sorted(ZEMBED_SUPPORTED_DIMS, reverse=True)
            raise ValueError(f"zembed dimensions must be one of {supported}")
        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            precision=precision,
            dimensions=dimensions,
            input_type="document",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self._projection_matrices: dict[str, np.ndarray] | None = None

    def _encode(self, model: Any, texts: list[str]) -> Any:
        """Encode with zembed prompts and apply learned projection if needed."""
        target_dim = self._dimensions or _ZEMBED_FULL_DIM
        original_precision = self._precision
        original_dimensions = self._dimensions

        self._precision = "float32"
        self._dimensions = None
        try:
            embeddings = super()._encode(model, texts)
        finally:
            self._precision = original_precision
            self._dimensions = original_dimensions

        if target_dim != _ZEMBED_FULL_DIM:
            embeddings = self._project(np.asarray(embeddings), target_dim)

        if original_precision != "float32":
            from sentence_transformers.util.quantization import (  # noqa: PLC0415
                quantize_embeddings,
            )

            embeddings = quantize_embeddings(
                embeddings,
                precision=cast(
                    Literal["float32", "int8", "uint8", "binary", "ubinary"],
                    original_precision,
                ),
            )
        return embeddings

    def _project(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Apply ZeroEntropy's learned projection and re-normalise vectors."""
        projection = self._projection_matrix(target_dim)
        reduced = embeddings @ projection
        norm = np.linalg.norm(reduced, axis=-1, keepdims=True)
        return reduced / np.where(norm == 0, 1, norm)

    def _projection_matrix(self, target_dim: int) -> np.ndarray:
        """Return the composed [2560, target_dim] zembed projection matrix."""
        mats = self._load_projection_matrices()
        dims = sorted(int(key) for key in mats)
        projection = mats[str(target_dim)]
        for upper_dim in dims[dims.index(target_dim) + 1 :]:
            projection = mats[str(upper_dim)] @ projection
        return projection

    def _load_projection_matrices(self) -> dict[str, np.ndarray]:
        """Download and cache zembed projection matrices from Hugging Face."""
        if self._projection_matrices is None:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            from safetensors.numpy import load_file  # noqa: PLC0415

            path = hf_hub_download(ZEMBED_MODEL_NAME, "projections.safetensors")
            self._projection_matrices = load_file(path)
        return self._projection_matrices
