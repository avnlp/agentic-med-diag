"""Parametrized streaming contract tests for every corpus loader.

Each param tuple carries a loader class, the expected ``doc.source``
literal, and a valid single-row dict.  ``astream`` is defined only on
``CorpusLoader`` (no loader overrides it), so every loader shares the
same streaming mechanics — batch-size cap, limit, offset, and source
provenance — proven here once with N=5 rows for each of the 5 loaders.

Tests use ``asyncio.run`` (not ``pytest-asyncio``) to avoid event-loop
socket issues with ``pytest-socket``, matching the existing ``*_astream.py``
pattern.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest

from am_diag.loaders.corpus.clinical_guidelines import ClinicalGuidelinesCorpusLoader
from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader
from am_diag.loaders.corpus.pubmed_case_reports import (
    PubmedCaseReportsCorpusLoader,
)
from am_diag.loaders.corpus.statpearls import StatPearlsCorpusLoader
from am_diag.loaders.corpus.textbooks import TextbooksCorpusLoader


_N_ROWS = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vary_rows(row_template: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Create ``n`` copies of ``row_template`` with unique identifiers."""
    rows: list[dict[str, Any]] = []
    for i in range(n):
        r = deepcopy(row_template)
        if "id" in row_template:
            r["id"] = f"{row_template['id']}_{i}"
        elif "pmcid" in row_template:
            r["pmcid"] = f"PMC{i}"
        rows.append(r)
    return rows


_PATCH_TARGET = "am_diag.loaders.corpus.base"

# ---------------------------------------------------------------------------
# Param list: (loader_cls, expected_source, valid_row, id)
# ---------------------------------------------------------------------------

_CORPUS_LOADERS = [
    pytest.param(
        PubMedCorpusLoader,
        "pubmed",
        {
            "id": "pubmed-12345678-0",
            "content": "Abstract text.",
            "title": "PubMed Article",
        },
        id="pubmed",
    ),
    pytest.param(
        StatPearlsCorpusLoader,
        "statpearls",
        {
            "id": "article-100024_2",
            "content": "Some text.",
            "title": "Chronic Total Occlusion -- Introduction",
        },
        id="statpearls",
    ),
    pytest.param(
        TextbooksCorpusLoader,
        "textbooks",
        {"id": "textbooks-Book_0", "content": "text", "title": "Book"},
        id="textbooks",
    ),
    pytest.param(
        ClinicalGuidelinesCorpusLoader,
        "clinical_guidelines",
        {
            "id": "g_0",
            "source": "WHO",
            "title": "T",
            "clean_text": "Clinical guideline text.",
            "url": None,
            "overview": "",
        },
        id="clinical_guidelines",
    ),
    pytest.param(
        PubmedCaseReportsCorpusLoader,
        "pubmed_case_reports",
        {
            "pmcid": "PMC123",
            "text": "Case report text.",
            "title": "Case Report",
            "journal": "NEJM",
            "publication_date": "2023-01-15",
        },
        id="pubmed_case_reports",
    ),
]


class TestCorpusStreamingContract:
    """Streaming contract enforced over every corpus loader.

    Tests batch-size cap, limit, offset, source literal, and load_sample.
    """

    # ------------------------------------------------------------------
    # batch-size cap
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "loader_cls,expected_source,row",
        _CORPUS_LOADERS,
    )
    def test_batch_size_cap(
        self,
        loader_cls,
        expected_source,
        row,
        patch_corpus_load_dataset,
    ):
        """Test that each yielded batch is at most batch_size items."""
        rows = _vary_rows(row, _N_ROWS)
        patch_corpus_load_dataset(_PATCH_TARGET, rows)
        loader = loader_cls()

        async def _run():
            batches: list[list[Any]] = []
            async for batch in loader.astream(batch_size=2):
                batches.append(batch)
            return batches

        batches = asyncio.run(_run())
        assert all(len(b) <= 2 for b in batches)
        assert sum(len(b) for b in batches) == _N_ROWS

    # ------------------------------------------------------------------
    # limit stops early
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "loader_cls,expected_source,row",
        _CORPUS_LOADERS,
    )
    def test_limit_stops_early(
        self,
        loader_cls,
        expected_source,
        row,
        patch_corpus_load_dataset,
    ):
        """Test that limit=N caps the total yielded Documents to N."""
        rows = _vary_rows(row, _N_ROWS)
        patch_corpus_load_dataset(_PATCH_TARGET, rows)
        loader = loader_cls()

        async def _run():
            chunks: list[Any] = []
            async for batch in loader.astream(batch_size=100, limit=3):
                chunks.extend(batch)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 3

    # ------------------------------------------------------------------
    # offset skips
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "loader_cls,expected_source,row",
        _CORPUS_LOADERS,
    )
    def test_offset_skips(
        self,
        loader_cls,
        expected_source,
        row,
        patch_corpus_load_dataset,
    ):
        """Test that offset=N skips N rows."""
        rows = _vary_rows(row, _N_ROWS)
        patch_corpus_load_dataset(_PATCH_TARGET, rows)
        loader = loader_cls()

        async def _run():
            chunks: list[Any] = []
            async for batch in loader.astream(batch_size=100, offset=2):
                chunks.extend(batch)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == _N_ROWS - 2

    # ------------------------------------------------------------------
    # source literal
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "loader_cls,expected_source,row",
        _CORPUS_LOADERS,
    )
    def test_source_literal(
        self,
        loader_cls,
        expected_source,
        row,
        patch_corpus_load_dataset,
    ):
        """Test that every yielded Document has the expected source literal."""
        rows = _vary_rows(row, _N_ROWS)
        patch_corpus_load_dataset(_PATCH_TARGET, rows)
        loader = loader_cls()

        async def _run():
            chunks: list[Any] = []
            async for batch in loader.astream(batch_size=100):
                chunks.extend(batch)
            return chunks

        chunks = asyncio.run(_run())
        assert all(d.source == expected_source for d in chunks)

    # ------------------------------------------------------------------
    # load_sample
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "loader_cls,expected_source,row",
        _CORPUS_LOADERS,
    )
    def test_load_sample_returns_n_documents(
        self,
        loader_cls,
        expected_source,
        row,
        patch_corpus_load_dataset,
    ):
        """Test that load_sample(n) returns exactly N Documents."""
        rows = _vary_rows(row, _N_ROWS)
        patch_corpus_load_dataset(_PATCH_TARGET, rows)
        loader = loader_cls()

        async def _run():
            return await loader.load_sample(n=4)

        chunks = asyncio.run(_run())
        assert len(chunks) == 4
