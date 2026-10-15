"""Tests for collapsing repeated node pairs into one edge with its assertions."""

from __future__ import annotations

import polars as pl
import pytest

from am_diag.loaders.graph.relations import collapse_relations
from am_diag.loaders.graph.vocab import Source


def _edges(
    *types: str, pair: tuple[str, str] = ("d1", "x1"), label: str = "DRG-DIS"
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "from": [pair[0]] * len(types),
            "to": [pair[1]] * len(types),
            "label": [label] * len(types),
            "type": list(types),
            "sources": [["DRUG_CENTRAL"]] * len(types),
        }
    )


class TestCollapseRelations:
    """One edge per pair: priority winner, kept assertions, and conflict flag."""

    @pytest.mark.parametrize(
        ("types", "winner"),
        [
            (["OFF_LABEL_USE", "CONTRAINDICATION", "INDICATION"], "INDICATION"),
            (["OFF_LABEL_USE", "CONTRAINDICATION"], "CONTRAINDICATION"),
            (["TARGET", "ENZYME", "CARRIER"], "ENZYME"),
            (["ENZYME", "INHIBITOR"], "INHIBITOR"),
            (["INHIBITOR", "ACTIVATOR"], "ACTIVATOR"),
            (["OTHER", "TARGET"], "TARGET"),
            (["NOT_A_KNOWN_RELATION", "OTHER"], "OTHER"),
        ],
    )
    def test_lowest_priority_number_wins(self, types: list[str], winner: str) -> None:
        """The relation with the lowest priority is the type; ties go alphabetical."""
        out = collapse_relations(_edges(*types), Source.DRUG_CENTRAL)
        assert out.height == 1
        assert out["type"].to_list() == [winner]

    def test_every_asserted_relation_is_kept_sorted(self) -> None:
        """Both the flat list and the source-tagged structs hold all relations."""
        out = collapse_relations(
            _edges("OFF_LABEL_USE", "INDICATION", "INDICATION"), Source.DRUG_CENTRAL
        ).row(0, named=True)
        assert out["asserted_relations"] == ["INDICATION", "OFF_LABEL_USE"]
        assert out["relation_assertions"] == [
            {"source": "DRUG_CENTRAL", "relation": "INDICATION"},
            {"source": "DRUG_CENTRAL", "relation": "OFF_LABEL_USE"},
        ]

    @pytest.mark.parametrize(
        ("types", "conflict"),
        [
            (["INDICATION", "CONTRAINDICATION"], True),
            (["INDICATION", "OFF_LABEL_USE"], True),
            (["ACTIVATOR", "INHIBITOR"], True),
            (["AGONIST", "ANTAGONIST"], True),
            (["INDICATION"], False),
            (["ENZYME", "TARGET"], False),
            (["INHIBITOR", "ENZYME"], False),
        ],
    )
    def test_conflict_needs_two_members_of_one_exclusive_group(
        self, types: list[str], conflict: bool
    ) -> None:
        """Only mutually exclusive relations on one pair raise the conflict flag."""
        out = collapse_relations(_edges(*types), Source.DRUG_CENTRAL)
        assert out["relation_conflict"].to_list() == [conflict]

    def test_pairs_are_independent_and_sorted(self) -> None:
        """Each pair resolves on its own and rows come back in key order."""
        edges = pl.concat(
            [
                _edges("CONTRAINDICATION", pair=("d2", "x1")),
                _edges("INDICATION", "CONTRAINDICATION", pair=("d1", "x1")),
            ]
        )
        out = collapse_relations(edges, Source.DRUG_CENTRAL)
        assert out["from"].to_list() == ["d1", "d2"]
        assert out["type"].to_list() == ["INDICATION", "CONTRAINDICATION"]
        assert out["relation_conflict"].to_list() == [True, False]

    def test_rows_are_ordered_by_from_then_to_then_label(self) -> None:
        """Pairs that share a head are ordered by tail, then by structural label."""
        shared_head = [
            ("d1", "x3", "DRG-PHE"),
            ("d1", "x1", "DRG-PHE"),
            ("d1", "x2", "DRG-DIS"),
            ("d1", "x3", "DRG-DIS"),
            ("d1", "x1", "DRG-DIS"),
            ("d1", "x2", "DRG-PHE"),
        ]
        edges = pl.concat(
            [
                _edges("INDICATION", pair=("d0", "x9")),
                *[
                    _edges("INDICATION", pair=(h, t), label=lb)
                    for h, t, lb in shared_head
                ],
            ]
        )
        out = collapse_relations(edges, Source.DRUG_CENTRAL)
        assert list(zip(out["from"], out["to"], out["label"], strict=True)) == [
            ("d0", "x9", "DRG-DIS"),
            ("d1", "x1", "DRG-DIS"),
            ("d1", "x1", "DRG-PHE"),
            ("d1", "x2", "DRG-DIS"),
            ("d1", "x2", "DRG-PHE"),
            ("d1", "x3", "DRG-DIS"),
            ("d1", "x3", "DRG-PHE"),
        ]

    def test_other_columns_are_carried_through(self) -> None:
        """Columns outside the pair keys and type survive the collapse."""
        out = collapse_relations(
            _edges("INDICATION", "OFF_LABEL_USE"), Source.DRUG_CENTRAL
        )
        assert out["sources"].to_list() == [["DRUG_CENTRAL"]]
        assert out["label"].to_list() == ["DRG-DIS"]
