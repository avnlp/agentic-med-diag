"""Unit tests for ZembedEmbedder with mocked SentenceTransformer/HF downloads."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from am_diag.vector.embedding.sentence_transformers import SentenceTransformersEmbedder
from am_diag.vector.embedding.zembed import (
    ZEMBED_MODEL_NAME,
    ZEMBED_SUPPORTED_DIMS,
    ZembedEmbedder,
)


def _make_zembed_embedder(
    dimensions: int = 2560,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
) -> ZembedEmbedder:
    """Build a ZembedEmbedder with valid zembed defaults and the given overrides."""
    return ZembedEmbedder(
        model_name=ZEMBED_MODEL_NAME,
        dimensions=dimensions,
        precision=precision,
    )


class TestZembedEmbedderInit:
    def test_accepts_each_supported_dimension(self):
        for dim in ZEMBED_SUPPORTED_DIMS:
            embedder = _make_zembed_embedder(dimensions=dim)
            assert embedder._dimensions == dim

    def test_rejects_unsupported_dimension(self):
        with pytest.raises(ValueError, match="zembed dimensions must be one of"):
            ZembedEmbedder(dimensions=999)

    def test_rejects_missing_dimension(self):
        with pytest.raises(ValueError, match="zembed dimensions must be one of"):
            ZembedEmbedder(dimensions=None)  # type: ignore[arg-type]


class TestZembedEmbedderEncode:
    def test_full_dimension_skips_projection_and_quantization(self):
        embedder = _make_zembed_embedder(dimensions=2560)
        full_vectors = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        with (
            patch.object(
                SentenceTransformersEmbedder,
                "_encode",
                return_value=full_vectors,
            ) as mock_parent,
            patch.object(ZembedEmbedder, "_project") as mock_project,
        ):
            result = embedder._encode(MagicMock(), ["text"])
        mock_parent.assert_called_once()
        mock_project.assert_not_called()
        np.testing.assert_array_equal(result, full_vectors)

    def test_lower_dimension_applies_projection(self):
        embedder = _make_zembed_embedder(dimensions=1280)
        full_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        projected = np.array([[0.5, 0.5]], dtype=np.float32)
        with (
            patch.object(
                SentenceTransformersEmbedder,
                "_encode",
                return_value=full_vectors,
            ),
            patch.object(
                ZembedEmbedder,
                "_project",
                return_value=projected,
            ) as mock_project,
        ):
            result = embedder._encode(MagicMock(), ["text"])
        mock_project.assert_called_once()
        call_args = mock_project.call_args.args
        np.testing.assert_array_equal(call_args[0], full_vectors)
        assert call_args[1] == 1280
        np.testing.assert_array_equal(result, projected)

    def test_temporarily_forces_float32_full_dim_and_restores_attrs(self):
        embedder = _make_zembed_embedder(dimensions=1280, precision="int8")
        captured: dict[str, object] = {}

        def fake_parent_encode(
            self_: SentenceTransformersEmbedder, _model: Any, _texts: list[str]
        ) -> Any:
            captured["precision"] = self_._precision
            captured["dimensions"] = self_._dimensions
            return np.array([[1.0, 0.0]], dtype=np.float32)

        with (
            patch.object(SentenceTransformersEmbedder, "_encode", fake_parent_encode),
            patch.object(ZembedEmbedder, "_project", side_effect=lambda emb, _dim: emb),
            patch(
                "sentence_transformers.util.quantization.quantize_embeddings",
                return_value=np.array([[1, 2]], dtype=np.int8),
            ),
        ):
            embedder._encode(MagicMock(), ["text"])

        assert captured == {"precision": "float32", "dimensions": None}
        assert embedder._precision == "int8"
        assert embedder._dimensions == 1280

    def test_quantizes_after_projection_for_non_float32_precision(self):
        embedder = _make_zembed_embedder(dimensions=1280, precision="int8")
        full_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        projected = np.array([[0.5, 0.5]], dtype=np.float32)
        quantized = np.array([[1, 2]], dtype=np.int8)
        with (
            patch.object(
                SentenceTransformersEmbedder,
                "_encode",
                return_value=full_vectors,
            ),
            patch.object(ZembedEmbedder, "_project", return_value=projected),
            patch(
                "sentence_transformers.util.quantization.quantize_embeddings",
                return_value=quantized,
            ) as mock_quantize,
        ):
            result = embedder._encode(MagicMock(), ["text"])
        mock_quantize.assert_called_once()
        call_args, call_kwargs = mock_quantize.call_args
        np.testing.assert_array_equal(call_args[0], projected)
        assert call_kwargs["precision"] == "int8"
        np.testing.assert_array_equal(result, quantized)

    def test_float32_precision_skips_quantization(self):
        embedder = _make_zembed_embedder(dimensions=2560, precision="float32")
        full_vectors = np.array([[0.1, 0.2]], dtype=np.float32)
        with (
            patch.object(
                SentenceTransformersEmbedder,
                "_encode",
                return_value=full_vectors,
            ),
            patch(
                "sentence_transformers.util.quantization.quantize_embeddings",
            ) as mock_quantize,
        ):
            embedder._encode(MagicMock(), ["text"])
        mock_quantize.assert_not_called()


class TestZembedEmbedderProject:
    def test_applies_projection_matrix(self):
        embedder = _make_zembed_embedder(dimensions=80)
        projection = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        with patch.object(
            ZembedEmbedder,
            "_projection_matrix",
            return_value=projection,
        ) as mock_matrix:
            result = embedder._project(embeddings, 80)
        mock_matrix.assert_called_once_with(80)
        np.testing.assert_allclose(result, [[1.0, 0.0]])

    def test_renormalizes_to_unit_length(self):
        embedder = _make_zembed_embedder(dimensions=80)
        projection = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        embeddings = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
        with patch.object(
            ZembedEmbedder,
            "_projection_matrix",
            return_value=projection,
        ):
            result = embedder._project(embeddings, 80)
        norms = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms, [1.0])

    def test_zero_vector_does_not_divide_by_zero(self):
        embedder = _make_zembed_embedder(dimensions=80)
        projection = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        embeddings = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        with patch.object(
            ZembedEmbedder,
            "_projection_matrix",
            return_value=projection,
        ):
            result = embedder._project(embeddings, 80)
        np.testing.assert_allclose(result, [[0.0, 0.0]])


class TestZembedEmbedderProjectionMatrix:
    def test_composes_matrices_from_target_up_to_full(self):
        embedder = _make_zembed_embedder(dimensions=80)
        mats = {
            "40": np.zeros((10, 40), dtype=np.float32),
            "80": np.ones((6, 80), dtype=np.float32),
            "160": np.full((3, 6), 2.0, dtype=np.float32),
        }
        embedder._projection_matrices = mats

        result = embedder._projection_matrix(80)

        expected = mats["160"] @ mats["80"]
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (3, 80)

    def test_full_dimension_key_requires_no_composition(self):
        embedder = _make_zembed_embedder(dimensions=40)
        mats = {"40": np.ones((5, 40), dtype=np.float32)}
        embedder._projection_matrices = mats

        result = embedder._projection_matrix(40)

        np.testing.assert_array_equal(result, mats["40"])


class TestZembedEmbedderLoadProjectionMatrices:
    def test_downloads_and_caches(self):
        embedder = _make_zembed_embedder(dimensions=1280)
        fake_mats = {"1280": np.eye(2, dtype=np.float32)}
        with (
            patch(
                "huggingface_hub.hf_hub_download",
                return_value="/tmp/fake.safetensors",
            ) as mock_download,
            patch("safetensors.numpy.load_file", return_value=fake_mats) as mock_load,
        ):
            first = embedder._load_projection_matrices()
            second = embedder._load_projection_matrices()

        assert first is fake_mats
        assert first is second
        mock_download.assert_called_once_with(
            ZEMBED_MODEL_NAME,
            "projections.safetensors",
        )
        mock_load.assert_called_once_with("/tmp/fake.safetensors")
