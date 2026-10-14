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


def _is_api(module: str) -> bool:
    """Return whether the dotted ``module`` name is ``am_diag.api`` or below it."""
    return module == "am_diag.api" or module.startswith("am_diag.api.")


def _from_base(node: ast.ImportFrom, package: list[str]) -> str:
    """Resolve the absolute dotted module that a ``from`` import names."""
    module = node.module or ""
    if node.level == 0:
        return module
    anchor = package[: len(package) - (node.level - 1)]
    return ".".join([*anchor, *([module] if module else [])])


def _imports_api(source: str, package: list[str]) -> bool:
    """Return whether ``source`` imports ``am_diag.api``, absolute or relative.

    Args:
        source: Python source text.
        package: Dotted package path of the module, for example
            ``["am_diag", "ingestion"]``, used to resolve relative imports.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            base = _from_base(node, package)
            if _is_api(base):
                return True
            if base == "am_diag" and any(alias.name == "api" for alias in node.names):
                return True
        if isinstance(node, ast.Import) and any(
            _is_api(alias.name) for alias in node.names
        ):
            return True
    return False


def _module_imports_api(path: Path) -> bool:
    """Return whether the module at ``path`` imports ``am_diag.api``."""
    package = ["am_diag", *path.relative_to(_PACKAGE_ROOT).parent.parts]
    return _imports_api(path.read_text("utf-8"), package)


class TestImportDetection:
    """The import scan recognizes every way of reaching ``am_diag.api``."""

    @pytest.mark.parametrize(
        "source",
        [
            "import am_diag.api",
            "from am_diag.api import Workspace",
            "from am_diag.api.graph import Graph",
            "from am_diag import api",
            "from ..api import Workspace",
            "from ..api.graph import Graph",
            "from .. import api",
        ],
    )
    def test_detects_api_import(self, source: str) -> None:
        """Absolute, relative, and package-attribute API imports are flagged."""
        assert _imports_api(source, ["am_diag", "ingestion"])

    @pytest.mark.parametrize(
        "source",
        [
            "from am_diag.common import api",
            "from . import cache",
            "from .jobs import manager",
            "from am_diag import version",
        ],
    )
    def test_ignores_unrelated_import(self, source: str) -> None:
        """Imports that do not reach ``am_diag.api`` are not flagged."""
        assert not _imports_api(source, ["am_diag", "ingestion"])


class TestEngineLayering:
    """Engine packages never depend on the public ``api`` facade."""

    @pytest.mark.parametrize(
        "module",
        _engine_modules(),
        ids=lambda p: str(p.relative_to(_PACKAGE_ROOT)),
    )
    def test_engine_module_does_not_import_api(self, module: Path) -> None:
        """A given engine module imports nothing from ``am_diag.api``."""
        assert not _module_imports_api(module), (
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
