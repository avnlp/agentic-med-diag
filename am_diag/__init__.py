"""Agentic GraphRAG for Medical Diagnosis."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from am_diag.version import __version__


if TYPE_CHECKING:
    from am_diag.api import Graph, Project, Workspace

_API_EXPORTS = frozenset({"Graph", "Project", "Workspace"})


def __getattr__(name: str) -> Any:
    """Load the API facade on first access so ``import am_diag`` stays light."""
    if name in _API_EXPORTS:
        return getattr(import_module("am_diag.api"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Graph", "Project", "Workspace", "__version__"]
