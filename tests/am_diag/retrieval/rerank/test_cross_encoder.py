"""Tests for cross-encoder reranker wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.rerank.cross_encoder import cross_encoder_rerank
from am_diag.vector.reranking.base import Reranker


class TestCrossEncoderRerank:
    async def test_empty_results_returns_empty(self) -> None:
        reranker = AsyncMock(spec=Reranker)
        result = await cross_encoder_rerank("query", [], reranker)
        assert result == []

    async def test_reranks_results(self) -> None:
        reranker = AsyncMock(spec=Reranker)
        reranker.rerank = AsyncMock(return_value=[])

        results = [
            SearchResult(
                item=Entity(id=uuid4(), name="metformin", label="Drug"),
                score=0.9,
                source="test",
            ),
        ]
        await cross_encoder_rerank("query", results, reranker)
        reranker.rerank.assert_awaited_once()

    async def test_maps_reranker_output_to_results(self) -> None:
        class _FakeRerankerResult:
            index: int = 0
            score: float = 0.95
            text: str = "test"

        reranker = AsyncMock(spec=Reranker)
        reranker.rerank = AsyncMock(return_value=[_FakeRerankerResult()])

        entity = Entity(id=uuid4(), name="metformin", label="Drug")
        results = [SearchResult(item=entity, score=0.5, source="test")]
        out = await cross_encoder_rerank("query", results, reranker)
        assert len(out) == 1
        assert out[0].score == 0.95
