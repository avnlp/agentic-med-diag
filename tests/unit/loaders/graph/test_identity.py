"""Tests for the deterministic id helpers used by structured-graph loaders.

These guarantee the union-by-MERGE property: the same CURIE (or the same edge
triple) always yields the same UUID, so re-ingestion and cross-loader overlap
collapse onto one node/edge instead of duplicating.
"""

from __future__ import annotations

from uuid import UUID

from am_diag.loaders.graph.identity import curie_to_entity_id, relation_id


class TestGraphIdentity:
    """Determinism and discrimination of CURIE-derived ids."""

    def test_curie_to_entity_id_is_deterministic(self) -> None:
        """The same CURIE always maps to the same entity id."""
        assert curie_to_entity_id("MONDO:0005148") == curie_to_entity_id(
            "MONDO:0005148"
        )

    def test_curie_to_entity_id_returns_uuid(self) -> None:
        """The helper returns a UUID instance."""
        assert isinstance(curie_to_entity_id("CHEMBL25"), UUID)

    def test_curie_to_entity_id_distinct_for_different_curies(self) -> None:
        """Different CURIEs map to different entity ids."""
        assert curie_to_entity_id("MONDO:0005148") != curie_to_entity_id("CHEMBL25")

    def test_curie_to_entity_id_strips_surrounding_whitespace(self) -> None:
        """Surrounding whitespace does not change the derived id."""
        assert curie_to_entity_id("  ENSG00000139618 ") == curie_to_entity_id(
            "ENSG00000139618"
        )

    def test_relation_id_is_deterministic(self) -> None:
        """The same (head, type, tail) triple yields the same relation id."""
        a = relation_id("CHEMBL25", "TARGET", "ENSG00000139618")
        b = relation_id("CHEMBL25", "TARGET", "ENSG00000139618")
        assert a == b

    def test_relation_id_is_order_sensitive(self) -> None:
        """Swapping head and tail produces a different relation id."""
        forward = relation_id("CHEMBL25", "TARGET", "ENSG00000139618")
        reversed_ = relation_id("ENSG00000139618", "TARGET", "CHEMBL25")
        assert forward != reversed_

    def test_relation_id_distinct_per_type(self) -> None:
        """The relation type participates in the derived id."""
        target = relation_id("CHEMBL25", "TARGET", "ENSG00000139618")
        enzyme = relation_id("CHEMBL25", "ENZYME", "ENSG00000139618")
        assert target != enzyme
