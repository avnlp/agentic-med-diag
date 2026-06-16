"""Cypher identifier validation helpers.

These functions are used by the loader and by the Neo4j client when building
queries with dynamic labels or relationship types (e.g. the secondary entity
type label). They prevent Cypher injection by validating against a strict regex
before back-tick substitution.
"""

from __future__ import annotations

import re


_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def validate_label(label: str) -> str:
    """Validate a Neo4j node label (PascalCase) and return it unchanged.

    Args:
        label: The node label string to validate.

    Returns:
        The label unchanged if valid.

    Raises:
        ValueError: For labels not matching [A-Z][A-Za-z0-9]*.
    """
    if not _LABEL_RE.match(label):
        raise ValueError(f"Invalid Neo4j label {label!r}: must be PascalCase")
    return label


def validate_rel_type(rel_type: str) -> str:
    """Validate a Neo4j relationship type (SCREAMING_SNAKE_CASE) and return it.

    Args:
        rel_type: The relationship type string to validate.

    Returns:
        The relationship type unchanged if valid.

    Raises:
        ValueError: For types not matching [A-Z][A-Z0-9_]*.
    """
    if not _REL_TYPE_RE.match(rel_type):
        raise ValueError(
            f"Invalid Neo4j rel type {rel_type!r}: must be SCREAMING_SNAKE_CASE"
        )
    return rel_type
