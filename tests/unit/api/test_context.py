"""Tests for context packing (am_diag/api/context.py)."""

from __future__ import annotations

from am_diag.api.context import pack_context
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.search import CypherRow, SearchResult


def _result(item) -> SearchResult:
    return SearchResult(item=item, score=1.0, source="test")


class TestPackContext:
    """Tests for pack_context."""

    def test_renders_cypher_rows_in_their_own_section(self) -> None:
        """Raw query rows are listed under a Query results heading."""
        results = [
            _result(Entity(name="metformin", label="Drug")),
            _result(CypherRow(values={"drug": "metformin", "n": 3})),
        ]

        packed = pack_context(results)

        assert "## Entities\n- metformin (Drug)" in packed
        assert "## Query results\n- drug: metformin, n: 3" in packed

    def test_cypher_row_round_trips_through_search_result(self) -> None:
        """A CypherRow survives SearchResult validation as itself."""
        row = CypherRow(values={"a": 1})

        assert _result(row).item == row
