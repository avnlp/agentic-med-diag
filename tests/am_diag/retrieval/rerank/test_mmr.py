"""Tests for MMR (Maximal Marginal Relevance) reranker."""

from __future__ import annotations

from uuid import uuid4

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import SearchResult
from am_diag.retrieval.rerank.mmr import maximal_marginal_relevance


def _result(name: str, score: float) -> SearchResult:
    return SearchResult(
        item=Entity(id=uuid4(), name=name, label="Drug"),
        score=score,
        source="test",
    )


class TestMMR:
    def test_empty_results_returns_empty(self) -> None:
        assert maximal_marginal_relevance([0.1, 0.2], []) == []

    def test_returns_all_results_when_top_k_not_set(self) -> None:
        results = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]
        out = maximal_marginal_relevance([0.1, 0.2], results)
        assert len(out) == 3

    def test_respects_top_k(self) -> None:
        results = [_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)]
        out = maximal_marginal_relevance([0.1, 0.2], results, top_k=2)
        assert len(out) == 2
