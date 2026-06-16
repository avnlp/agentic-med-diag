"""Tests for RRF (Reciprocal Rank Fusion)."""

from __future__ import annotations

from uuid import uuid4

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.rerank.rrf import rrf


_METFORMIN = Entity(id=uuid4(), name="metformin", label="Drug")
_DIABETES = Entity(id=uuid4(), name="diabetes", label="Disease")
_ASPIRIN = Entity(id=uuid4(), name="aspirin", label="Drug")


class TestRRF:
    def test_empty_lists_returns_empty(self) -> None:
        assert rrf([]) == []

    def test_single_list_returns_same_order(self) -> None:
        results = [
            SearchResult(item=_METFORMIN, score=0.9, source="test"),
            SearchResult(item=_DIABETES, score=0.8, source="test"),
        ]
        fused = rrf([results])
        assert len(fused) == 2
        assert fused[0].score >= fused[1].score

    def test_fuses_two_lists_with_same_item(self) -> None:
        """Same item appearing in two lists is deduped with combined score."""
        r1 = SearchResult(item=_METFORMIN, score=0.9, source="entity")
        r2 = SearchResult(item=_METFORMIN, score=0.8, source="relation")
        fused = rrf([[r1], [r2]])
        assert len(fused) == 1  # deduped by item id
        # Fused score = 1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(fused[0].score - expected) < 1e-6

    def test_higher_ranked_items_get_higher_score(self) -> None:
        r1 = SearchResult(item=_METFORMIN, score=0.9, source="e1")
        r2 = SearchResult(item=_DIABETES, score=0.8, source="e1")
        r3 = SearchResult(item=_METFORMIN, score=0.9, source="e2")
        r4 = SearchResult(item=_ASPIRIN, score=0.7, source="e2")
        fused = rrf([[r1, r2], [r3, r4]])
        metformin_score = 2.0 / 61.0
        diabetes_score = 1.0 / 62.0
        aspirin_score = 1.0 / 62.0
        assert abs(fused[0].score - metformin_score) < 1e-6
        assert fused[1].score in (diabetes_score, aspirin_score)

    def test_custom_k_value(self) -> None:
        r1 = SearchResult(item=_METFORMIN, score=0.9, source="e1")
        r2 = SearchResult(item=_METFORMIN, score=0.9, source="e2")
        fused = rrf([[r1], [r2]], k=10)
        expected = 2.0 / 11.0
        assert abs(fused[0].score - expected) < 1e-6
