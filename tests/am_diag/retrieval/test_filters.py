"""Tests for SearchFilters — typed metadata filtering for vector and graph searches."""

from __future__ import annotations

from am_diag.retrieval.filters import SearchFilters


class TestSearchFiltersToPayloadFilter:
    def test_none_fields_returns_none(self) -> None:
        filters = SearchFilters()
        assert filters.to_payload_filter() is None

    def test_single_label_produces_scalar(self) -> None:
        filters = SearchFilters(labels=["Drug"])
        result = filters.to_payload_filter()
        assert result == {"label": "Drug"}

    def test_multiple_labels_produces_list(self) -> None:
        filters = SearchFilters(labels=["Drug", "Disease"])
        result = filters.to_payload_filter()
        assert result == {"label": ["Drug", "Disease"]}

    def test_relation_types(self) -> None:
        filters = SearchFilters(relation_types=["TREATED_BY"])
        result = filters.to_payload_filter()
        assert result == {"type": "TREATED_BY"}

    def test_all_fields_combined(self) -> None:
        filters = SearchFilters(
            labels=["Drug"],
            relation_types=["TREATED_BY", "CAUSES"],
            document_sources=["pubmed"],
            community_level=1,
            properties={"verified": True},
        )
        result = filters.to_payload_filter()
        assert result is not None
        assert result["label"] == "Drug"
        assert result["type"] == ["TREATED_BY", "CAUSES"]
        assert result["document_source"] == "pubmed"
        assert result["level"] == 1
        assert result["verified"] is True


class TestSearchFiltersToCypherWhere:
    def test_no_filters_returns_empty(self) -> None:
        filters = SearchFilters()
        clause, params = filters.to_cypher_where()
        assert clause == ""
        assert params == {}

    def test_labels_use_label_predicate(self) -> None:
        filters = SearchFilters(labels=["Drug", "Disease"])
        clause, params = filters.to_cypher_where()
        assert "n:Drug" in clause
        assert "n:Disease" in clause
        assert params == {}

    def test_relation_types(self) -> None:
        filters = SearchFilters(relation_types=["TREATED_BY"])
        clause, params = filters.to_cypher_where()
        assert "type(n)" in clause
        assert "$relation_types" in clause
        assert params["relation_types"] == ["TREATED_BY"]

    def test_document_sources(self) -> None:
        filters = SearchFilters(document_sources=["pubmed"])
        clause, params = filters.to_cypher_where()
        assert "document_source" in clause
        assert "$document_sources" in clause
        assert params["document_sources"] == ["pubmed"]

    def test_community_level(self) -> None:
        filters = SearchFilters(community_level=2)
        clause, params = filters.to_cypher_where()
        assert "level" in clause
        assert "$community_level" in clause
        assert params["community_level"] == 2

    def test_custom_var_name(self) -> None:
        filters = SearchFilters(community_level=1)
        clause, params = filters.to_cypher_where(var="entity")
        assert "entity.level" in clause
        assert "$community_level" in clause

    def test_properties(self) -> None:
        filters = SearchFilters(properties={"verified": True})
        clause, params = filters.to_cypher_where()
        assert "n.verified" in clause
        assert params["verified"] is True
