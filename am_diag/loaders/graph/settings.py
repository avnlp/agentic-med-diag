"""Filesystem locations for raw structured-graph source data."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphSourcesSettings(BaseSettings):
    """Root paths for raw per-dataset source files and shared reference data.

    Each loader resolves its inputs under ``raw_data_dir / <loader_name>``; the
    shared reference layer (Open Targets, HGNC, ontologies) reads from
    ``reference_dir``. All fields can be overridden via the ``GRAPH_`` env prefix
    or supplied directly as constructor arguments.

    Attributes:
        raw_data_dir: Root holding one subdirectory per source.
            Env: ``GRAPH_RAW_DATA_DIR``.
        reference_dir: Root holding the shared reference inputs (``opentargets/``,
            ``gene_names/``, ``ontology/``). Env: ``GRAPH_REFERENCE_DIR``.
    """

    model_config = SettingsConfigDict(
        env_prefix="GRAPH_",
        env_file=".env",
        extra="ignore",
    )

    raw_data_dir: Path = Field(default=Path("data/graph/raw"))
    reference_dir: Path = Field(default=Path("data/graph/reference"))

    def source_dir(self, loader_name: str) -> Path:
        """Return the raw directory for one loader (``raw_data_dir / loader_name``)."""
        return self.raw_data_dir / loader_name
