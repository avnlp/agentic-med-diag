"""Collapse several relations on one node pair into a single edge.

OptimusKG keeps one edge per ``(from, to, label)``. When sources disagree, the edge's
``type`` is the relation with the lowest priority number, and every asserted relation
stays on the edge so no statement is lost. A drug and a disease asserted as both an
``INDICATION`` and a ``CONTRAINDICATION`` is flagged with ``relation_conflict``. The
shared ``collapse_relations`` function applies that rule to one source's edges; it can
only see the statements of the source it is given.
"""

from __future__ import annotations

import polars as pl

from am_diag.common.schema.opentargets import DRUG_GENE_ACTIONS
from am_diag.loaders.graph.vocab import Relation, Source


# Lower numbers win. Pharmacological actions are the most specific drug-gene
# statements, then the DrugBank roles, and OTHER is the least specific.
RELATION_PRIORITY: dict[str, int] = {
    Relation.INDICATION.value: 1,
    Relation.CONTRAINDICATION.value: 2,
    Relation.OFF_LABEL_USE.value: 3,
    **dict.fromkeys(DRUG_GENE_ACTIONS, 1),
    Relation.ENZYME.value: 10,
    Relation.TRANSPORTER.value: 11,
    Relation.CARRIER.value: 12,
    Relation.TARGET.value: 13,
    Relation.OTHER.value: 100,
}

_UNKNOWN_PRIORITY = 999

# Groups of relations that cannot all be true of the same node pair.
_MUTUALLY_EXCLUSIVE: tuple[frozenset[str], ...] = (
    frozenset(
        {
            Relation.INDICATION.value,
            Relation.CONTRAINDICATION.value,
            Relation.OFF_LABEL_USE.value,
        }
    ),
    frozenset({"AGONIST", "ANTAGONIST", "INVERSE_AGONIST"}),
    frozenset({"ACTIVATOR", "INHIBITOR"}),
    frozenset({"POSITIVE_MODULATOR", "NEGATIVE_MODULATOR"}),
    frozenset({"POSITIVE_ALLOSTERIC_MODULATOR", "NEGATIVE_ALLOSTERIC_MODULATOR"}),
)

_PAIR_KEYS = ["from", "to", "label"]


def collapse_relations(edges: pl.DataFrame, source: Source) -> pl.DataFrame:
    """Return one edge per ``(from, to, label)`` from edges that may repeat a pair.

    Args:
        edges: Edges with ``from``, ``to``, ``label``, and ``type`` columns, one row
            per asserted relation. Every other column must hold the same value for
            all rows of a pair; the first value is kept.
        source: The dataset that made these assertions.

    Returns:
        One row per pair. ``type`` is the relation with the lowest priority number,
        with ties broken alphabetically and relations missing from
        ``RELATION_PRIORITY`` ranked last. The frame gains ``asserted_relations`` (the
        distinct relations, sorted), ``relation_assertions`` (``{source, relation}``
        structs), and ``relation_conflict`` (true when two members of a mutually
        exclusive group were asserted). Rows are sorted by the pair keys.
    """
    others = [c for c in edges.columns if c not in {*_PAIR_KEYS, "type"}]
    priority = pl.col("type").replace_strict(
        RELATION_PRIORITY, default=_UNKNOWN_PRIORITY, return_dtype=pl.Int64
    )
    conflicts = [
        pl.col("asserted_relations")
        .list.eval(pl.element().is_in(sorted(group)).cast(pl.UInt32))
        .list.sum()
        >= 2
        for group in _MUTUALLY_EXCLUSIVE
    ]
    return (
        edges.group_by(_PAIR_KEYS)
        .agg(
            pl.col("type").sort_by([priority, pl.col("type")]).first(),
            pl.col("type").unique().sort().alias("asserted_relations"),
            *[pl.col(c).first() for c in others],
        )
        .with_columns(
            pl.col("asserted_relations")
            .list.eval(
                pl.struct(
                    pl.lit(source.value).alias("source"),
                    pl.element().alias("relation"),
                )
            )
            .alias("relation_assertions"),
            pl.any_horizontal(conflicts).alias("relation_conflict"),
        )
        .select(
            [*_PAIR_KEYS, "type", *others, "asserted_relations"]
            + ["relation_assertions", "relation_conflict"]
        )
        .sort(_PAIR_KEYS)
    )
