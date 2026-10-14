"""Shared test fixtures for corpus loader unit tests.

Provides fixtures for creating in-memory HuggingFace IterableDatasets
and patching ``load_dataset`` so corpus loaders read from local data
instead of the network.
"""

from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset


@pytest.fixture
def make_streaming_dataset():
    """Create a real ``IterableDataset`` from a list of dict rows.

    Usage in a test::

        def test_something(self, make_streaming_dataset):
            ds = make_streaming_dataset([{"id": "1", ...}, ...])
    """

    def _make(rows: list[dict[str, Any]]):
        return Dataset.from_list(rows).to_iterable_dataset()

    return _make


@pytest.fixture
def patch_corpus_load_dataset(mocker, make_streaming_dataset):
    """Patch ``load_dataset`` in a given module path to return mock rows.

    The returned callable takes a ``module_path`` and a list of row dicts,
    then patches ``<module_path>.load_dataset`` to return the supplied
    rows as an ``IterableDataset``.
    """

    def _patch(module_path: str, rows: list[dict[str, Any]]):
        def _factory(*args: Any, **kwargs: Any):  # noqa: ARG001
            return make_streaming_dataset(rows)

        mocker.patch(f"{module_path}.load_dataset", side_effect=_factory)

    return _patch
