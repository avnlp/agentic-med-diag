"""Text normalization utilities (NFKC, whitespace, lowercase, singularization)."""

from __future__ import annotations

import re
import unicodedata

import inflect


def normalize_text(text: str) -> str:
    """NFKC-normalize, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def singularize(text: str) -> str:
    """Simple English singularization (inflect-based, fallback to text)."""
    p = inflect.engine()
    singular = p.singular_noun(text)
    return singular if singular else text


__all__ = ["normalize_text", "singularize"]
