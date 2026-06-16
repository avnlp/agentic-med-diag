"""Tests for node distance graph-aware reranker."""

from __future__ import annotations

from uuid import uuid4

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.rerank.node_distance import node_distance_reranker


def _result(name: str, score: float, matched_on: str | None = None) -> SearchResult:
    return SearchResult(
        item=Entity(id=uuid4(), name=name, label="Drug"),
        score=score,
        source="bfs",
        matched_on=matched_on,
    )


class TestNodeDistanceReranker:
    def test_no_hop_results_unchanged(self) -> None:
        results = [_result("a", 0.9), _result("b", 0.8)]
        out = node_distance_reranker("center", results)
        assert len(out) == 2
        assert abs(out[0].score - 0.9) < 1e-6
        assert abs(out[1].score - 0.8) < 1e-6

    def test_closer_hops_get_higher_boost(self) -> None:
        results = [
            _result("a", 1.0, matched_on="hop_1"),
            _result("b", 1.0, matched_on="hop_2"),
        ]
        out = node_distance_reranker("center", results, decay=0.5)
        assert out[0].score > out[1].score
        assert abs(out[0].score - 0.5) < 1e-6
        assert abs(out[1].score - 0.25) < 1e-6

    def test_custom_decay(self) -> None:
        results = [_result("a", 1.0, matched_on="hop_1")]
        out = node_distance_reranker("center", results, decay=0.3)
        assert abs(out[0].score - 0.3) < 1e-6
