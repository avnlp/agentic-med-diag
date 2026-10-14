"""Tests for job, connection, and graph-record validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from am_diag.common.data_models.connection import VectorConnection
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.job import JobConfig, JobRecord


class TestJobRecord:
    """Tests for JobRecord and JobConfig validation."""

    @pytest.mark.parametrize("job_id", ["../x", "a/b", "", "a b", ".."])
    def test_rejects_job_ids_that_are_not_one_path_component(self, job_id):
        """Verify unsafe job ids are refused."""
        with pytest.raises(ValidationError):
            JobRecord(job_id=job_id, kind="ingestion", graph_id="g")

    def test_ingestion_requires_graph_id(self):
        """Verify an ingestion job without a graph is rejected."""
        with pytest.raises(ValidationError, match="graph_id"):
            JobRecord(job_id="j", kind="ingestion")

    def test_qa_requires_project_id(self):
        """Verify a QA job without a project is rejected."""
        with pytest.raises(ValidationError, match="project_id"):
            JobRecord(job_id="j", kind="qa")

    @pytest.mark.parametrize(
        "field",
        [
            "extract_concurrency",
            "embed_batch_size",
            "write_batch_size",
            "max_retries",
            "answer_concurrency",
        ],
    )
    def test_config_limits_must_be_positive(self, field):
        """Verify zero limits that could hang or skip work are rejected."""
        with pytest.raises(ValidationError):
            JobConfig(**{field: 0})


class TestGraphRecord:
    """Tests for GraphRecord and VectorConnection defaults."""

    def test_partial_collections_are_filled_with_defaults(self):
        """Verify missing collection kinds fall back to their defaults."""
        conn = VectorConnection(url_env="URL", collections={"entity": "custom"})
        assert conn.collections["entity"] == "custom"
        assert conn.collections["chunk"] == "am_diag_chunk"
        assert set(conn.collections) == {"entity", "relation", "chunk", "community"}

    def test_default_schema_is_not_shared_between_records(self):
        """Verify mutating one record's schema leaves other records untouched."""
        kwargs = {
            "graph_id": "g",
            "neo4j": {"uri": "bolt://h", "user_env": "U", "password_env": "P"},
            "vector": {"url_env": "URL"},
        }
        first = GraphRecord(**kwargs)
        second = GraphRecord(**kwargs)
        first.graph_schema.entity_types.clear()
        assert second.graph_schema.entity_types
