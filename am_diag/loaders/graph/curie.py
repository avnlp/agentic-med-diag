"""OptimusKG canonical CURIE conventions (shared across loaders).

Adopting OptimusKG's exact scheme means raw-built nodes share identity with the
released-gold loader (same ``curie_to_entity_id``), so the two loader families
``MERGE`` into the same Neo4j nodes and the released gold doubles as a correctness
oracle. These conventions live here once so loaders and the ``Harmonizer`` agree.
"""

from __future__ import annotations

import polars as pl


# OBO ontologies use Biolink-style CURIEs with ``_`` instead of ``:``.
_OBO_COLON_PREFIXES = ("UBERON", "HP", "MONDO", "GO", "CL", "Orphanet")


def to_biolink_curie(expr: pl.Expr) -> pl.Expr:
    """Rewrite ``PREFIX:1234`` to ``PREFIX_1234`` for OBO prefixes.

    Args:
        expr: A string column expression holding OBO CURIEs.

    Returns:
        An expression with the leading ``PREFIX:`` rewritten to ``PREFIX_``.
    """
    out = expr
    for prefix in _OBO_COLON_PREFIXES:
        out = out.str.replace(f"^{prefix}:", f"{prefix}_")
    return out


def prefixed(expr: pl.Expr, prefix: str) -> pl.Expr:
    """Prepend ``prefix`` to ids that do not already carry it.

    Args:
        expr: A string column expression of bare or already-prefixed ids.
        prefix: The CURIE prefix to ensure, e.g. ``"NCBIGene:"``.

    Returns:
        An expression where every value starts with ``prefix`` exactly once.
    """
    return (
        pl.when(expr.str.starts_with(prefix))
        .then(expr)
        .otherwise(pl.concat_str([pl.lit(prefix), expr]))
    )
