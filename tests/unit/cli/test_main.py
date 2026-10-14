"""Tests for the am-diag CLI argument parser and source resolution.

Verifies the subcommand grammar parses and that ``corpus:``/``loader:`` source
specs resolve against the relocated registries. No commands are executed.
"""

from __future__ import annotations

import pytest

from am_diag.cli.main import _build_source, build_parser
from am_diag.cli.registries import (
    CORPUS_LOADER_REGISTRY,
    DATASET_REGISTRY,
    GRAPH_LOADER_REGISTRY,
)
from am_diag.loaders.corpus.pubmed import PubMedCorpusLoader
from am_diag.loaders.graph.drugbank import DrugBankGraphLoader
from am_diag.loaders.graph.gene_ontology import GeneOntologyGraphLoader


class TestParser:
    """Tests for the CLI parser grammar."""

    def test_graphs_create_parses(self):
        """Verify graphs create parses required arguments."""
        args = build_parser().parse_args(
            ["graphs", "create", "--graph-id", "med", "--neo4j-uri", "bolt://h:7687"]
        )
        assert args.command == "graphs"
        assert args.action == "create"
        assert args.graph_id == "med"

    def test_ingest_collects_multiple_sources(self):
        """Verify --source is repeatable and --finalize is a flag."""
        args = build_parser().parse_args(
            [
                "ingest",
                "--graph",
                "med",
                "--source",
                "corpus:pubmed",
                "--source",
                "loader:drugbank:/data/drugbank",
                "--reference-dir",
                "/data/reference",
                "--finalize",
            ]
        )
        assert args.source == ["corpus:pubmed", "loader:drugbank:/data/drugbank"]
        assert args.reference_dir == "/data/reference"
        assert args.finalize is True

    def test_search_takes_positional_query(self):
        """Verify search accepts a project and a positional query."""
        args = build_parser().parse_args(
            ["search", "--project", "clinical", "sepsis management"]
        )
        assert args.project == "clinical"
        assert args.query == "sepsis management"

    def test_jobs_resume_parses(self):
        """Verify jobs resume parses the job id."""
        args = build_parser().parse_args(["jobs", "resume", "--job-id", "abc"])
        assert args.action == "resume"
        assert args.job_id == "abc"

    def test_missing_subcommand_errors(self):
        """Verify a missing subcommand exits with an error."""
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestRegistries:
    """Tests for the relocated loader/dataset registries."""

    def test_registries_are_populated(self):
        """Verify the three registries expose the expected loaders."""
        assert "pubmed" in CORPUS_LOADER_REGISTRY
        assert "drugbank" in GRAPH_LOADER_REGISTRY
        assert "medqa" in DATASET_REGISTRY


class TestBuildSource:
    """Tests for resolving --source specs into loader instances."""

    def test_corpus_spec_builds_corpus_loader(self):
        """Verify a corpus spec resolves to its loader without extra arguments."""
        assert isinstance(_build_source("corpus:pubmed"), PubMedCorpusLoader)

    def test_graph_loader_spec_takes_a_source_dir(self):
        """Verify a graph loader that needs only a directory is built from the spec."""
        source = _build_source("loader:gene_ontology:/data/go")

        assert isinstance(source, GeneOntologyGraphLoader)

    def test_graph_loader_with_shared_services_uses_reference_dir(self):
        """Verify harmonizer and enricher are built from the reference directory."""
        source = _build_source(
            "loader:drugbank:/data/drugbank", reference_dir="/data/reference"
        )

        assert isinstance(source, DrugBankGraphLoader)

    def test_graph_loader_needing_reference_without_it_exits(self):
        """Verify a missing reference directory is reported, not a TypeError."""
        with pytest.raises(SystemExit, match="--reference-dir"):
            _build_source("loader:drugbank:/data/drugbank")

    def test_graph_loader_without_source_dir_exits(self):
        """Verify a loader spec with no directory is rejected with a clear message."""
        with pytest.raises(SystemExit, match="source directory"):
            _build_source("loader:gene_ontology")

    @pytest.mark.parametrize("spec", ["pubmed", "corpus:nope", "loader:nope:/x"])
    def test_unknown_spec_exits(self, spec):
        """Verify malformed or unknown specs exit with the offending spec."""
        with pytest.raises(SystemExit, match="Unknown source spec"):
            _build_source(spec)
