"""Shared test fixtures for dataset loader unit tests.

The two fixtures here let test classes create a real HuggingFace
``IterableDataset`` from a list of dicts and then patch
``load_dataset`` so the loader under test reads from that in-memory
data instead of the network.
"""

from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset


@pytest.fixture
def make_iterable_dataset():
    """Create a real ``IterableDataset`` from a list of dict rows.

    Usage in a test::

        def test_something(self, make_iterable_dataset):
            ds = make_iterable_dataset([{"col": "val"}, ...])
            # ds is a true datasets IterableDataset
    """

    def _make(rows: list[dict[str, Any]]):
        return Dataset.from_list(rows).to_iterable_dataset()

    return _make


@pytest.fixture
def patch_load_dataset(mocker, make_iterable_dataset):
    """Patch ``load_dataset`` in a given module path to return mock rows.

    The returned callable takes a ``module_path`` and a list of row dicts,
    then patches ``<module_path>.load_dataset`` to return the supplied
    rows as an ``IterableDataset`` regardless of the arguments passed to
    ``load_dataset``.

    Usage in a test::

        def test_something(self, patch_load_dataset):
            patch_load_dataset(
                "am_diag.loaders.dataset.medqa",
                [{"question": "...", ...}],
            )
            samples = MedQADataset().load()
    """

    def _patch(module_path: str, rows: list[dict[str, Any]]):
        def _factory(*args: Any, **kwargs: Any):  # noqa: ARG001
            return make_iterable_dataset(rows)

        mocker.patch(f"{module_path}.load_dataset", side_effect=_factory)

    return _patch
