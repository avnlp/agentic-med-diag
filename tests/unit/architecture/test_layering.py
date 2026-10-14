"""Architecture guard: the engines must not import the ``api`` facade.

Dependencies point inward: ``ingestion``, ``agents``, and
``retrieval`` depend on consumer-owned Protocols (``IngestionTarget``,
``Answerer``, ``SearchProvider``), never on ``am_diag.api``. This test fails if a
future change introduces an upward import.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import am_diag
from am_diag.ingestion.jobs.manager import JobManager


_ENGINE_PACKAGES = ("ingestion", "agents", "retrieval")
_PACKAGE_ROOT = Path(am_diag.__file__).parent


def _engine_modules() -> list[Path]:
    """Return every ``.py`` file under the engine packages."""
    modules: list[Path] = []
    for package in _ENGINE_PACKAGES:
        modules.extend(sorted((_PACKAGE_ROOT / package).rglob("*.py")))
    return modules


def _imports_api(path: Path) -> bool:
    """Return whether ``path`` imports anything under ``am_diag.api``."""
    tree = ast.parse(path.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "am_diag.api"
        ):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.startswith("am_diag.api") for alias in node.names
        ):
            return True
    return False


class TestEngineLayering:
    """Engine packages never depend on the public ``api`` facade."""

    @pytest.mark.parametrize(
        "module",
        _engine_modules(),
        ids=lambda p: str(p.relative_to(_PACKAGE_ROOT)),
    )
    def test_engine_module_does_not_import_api(self, module: Path) -> None:
        """A given engine module imports nothing from ``am_diag.api``."""
        assert not _imports_api(module), (
            f"{module.relative_to(_PACKAGE_ROOT)} imports am_diag.api; engines must "
            "not depend on the facade. Depend on an ingestion-owned "
            "Protocol instead."
        )


class TestJobManagerDecoupled:
    """JobManager is constructed without a Workspace/client back-reference."""

    def test_job_manager_takes_no_workspace_client(self) -> None:
        """JobManager.__init__ exposes no client/workspace parameter.

        The import scan above cannot catch a duck-typed coupling (the old
        ``client: Any`` back-reference), so this asserts the constructor shape
        directly.
        """
        params = set(inspect.signature(JobManager.__init__).parameters)
        assert "client" not in params
        assert "workspace" not in params
