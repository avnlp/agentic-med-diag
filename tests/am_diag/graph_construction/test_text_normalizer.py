"""Tests for am_diag/graph_construction/resolve/text.py."""

from __future__ import annotations

from am_diag.graph_construction import normalize_text


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Metformin") == "metformin"

    def test_collapses_whitespace(self):
        assert (
            normalize_text("type  2\tdiabetes\nmellitus") == "type 2 diabetes mellitus"
        )

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  metformin  ") == "metformin"

    def test_nfkc_normalizes_unicode(self):
        # U+FB00 LATIN SMALL LIGATURE FF NFKC-decomposes to "ff".
        assert normalize_text("ﬀee") == "ffee"

    def test_idempotent(self):
        text = "  Metformin   Treats   T2DM  "
        once = normalize_text(text)
        assert normalize_text(once) == once
