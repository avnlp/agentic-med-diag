"""Cached .cypher file loader with safe identifier substitution.

Cypher files are loaded once and cached (lru_cache). Dynamic identifiers
(labels, relationship types) are substituted via ${key} placeholders after
validation against a strict regex. Values are never interpolated — they
always use $param placeholders.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


_ROOT = Path(__file__).parent
_SAFE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def safe_identifier(value: str) -> str:
    """Validate and backtick-escape a Cypher label or relationship type name.

    Raises ValueError for identifiers that don't match [A-Za-z][A-Za-z0-9_]*.
    Backtick-escaping makes reserved words safe and is idempotent for normal names.

    Args:
        value: The identifier string to validate and escape.

    Returns:
        The identifier wrapped in backticks, e.g. ``Disease`` -> `` `Disease` ``.

    Raises:
        ValueError: If the identifier contains unsafe characters.
    """
    if not _SAFE_RE.match(value):
        raise ValueError(
            f"Unsafe Cypher identifier {value!r}: must match [A-Za-z][A-Za-z0-9_]*"
        )
    return f"`{value}`"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Load a Cypher file by name, cached after first read.

    Args:
        name: Path relative to this package, without .cypher suffix.
              E.g. "ingest/upsert_entities", "schema/constraints".

    Returns:
        The file's text content.

    Raises:
        FileNotFoundError: If no .cypher file exists at the given path.
    """
    path = _ROOT / f"{name}.cypher"
    return path.read_text("utf-8")


def render(name: str, **identifiers: str) -> str:
    """Load a Cypher file and substitute dynamic identifier placeholders.

    Placeholder format: ${key} in the .cypher file. Each key in `identifiers`
    must be a valid Cypher identifier (validated + backtick-escaped). Values
    intended for Cypher parameters still use $param syntax and are NOT touched.

    Args:
        name: .cypher file name (passed to load()).
        **identifiers: Label/type names to substitute for ${key} placeholders.

    Returns:
        The Cypher string with ${key} substituted.

    Raises:
        ValueError: If any identifier fails the safety check.
    """
    query = load(name)
    for key, value in identifiers.items():
        query = query.replace(f"${{{key}}}", safe_identifier(value))
    return query
