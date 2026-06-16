"""Metadata descriptor type for DataPoint-based models.

`MetaData` is a TypedDict that describes how a `DataPoint` subclass
should be indexed and deduplicated:

* `type` — Optional discriminator string (reserved for future use).
* `index_fields` — Field names whose values are embedded in the vector
  database.
* `identity_fields` — Field names whose values collectively form the
  deterministic UUID5 identity key for deduplication.
"""

from __future__ import annotations

from typing_extensions import NotRequired, TypedDict


class MetaData(TypedDict):
    """Metadata descriptor carried by every DataPoint.

    `index_fields` drives embedding; `identity_fields` drives
    deterministic UUID5 generation.
    """

    type: NotRequired[str]
    index_fields: list[str]
    identity_fields: NotRequired[list[str]]
