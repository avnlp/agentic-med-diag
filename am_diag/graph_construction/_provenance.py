"""Tolerate optional provenance on extraction data points.

``Entity``/``Relation`` carry ``provenance`` as ``Provenance | None`` so structured
graph loaders need not synthesize extraction metadata. Text extractors always set
it, so the merge components read it through ``ensure_provenance`` to narrow the
optional away (creating an empty record only in the never-hit absent case).
"""

from __future__ import annotations

from typing import Protocol

from am_diag.common.data_models.provenance import Provenance


class _HasProvenance(Protocol):
    """Anything with a mutable optional ``provenance`` attribute."""

    provenance: Provenance | None


def ensure_provenance(data_point: _HasProvenance) -> Provenance:
    """Return the data point's provenance, attaching an empty one if absent.

    Args:
        data_point: An entity or relation with an optional ``provenance``.

    Returns:
        The non-``None`` ``Provenance`` (newly attached if it was ``None``).
    """
    prov = data_point.provenance
    if prov is None:
        prov = Provenance()
        data_point.provenance = prov
    return prov
