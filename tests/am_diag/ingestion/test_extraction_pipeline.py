"""Tests for am_diag/ingestion/extraction_pipeline.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from am_diag.common.data_models import Document, Entity, Relation
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.schema import EntityType, GraphSchema
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.ingestion.extraction_pipeline import build_extraction_pipeline, prepare
from am_diag.ingestion.models import ExtractionReport, IngestionReport


pytestmark = pytest.mark.enable_socket


def _no_extractor_config(tmp_path, **overrides) -> ExtractionPipelineConfig:
    return ExtractionPipelineConfig(
        use_gliner=False,
        use_glirel=False,
        use_llm=False,
        output_dir=str(tmp_path),
        **overrides,
    )


def _mock_neo4j_client() -> MagicMock:
    client = MagicMock()
    client.execute_query = AsyncMock(return_value=[])
    client.execute_batch = AsyncMock(return_value=0)
    client.setup_schema = AsyncMock()
    client.ingest_entities = AsyncMock(return_value=0)
    client.ingest_relations = AsyncMock(return_value=0)
    client.close = AsyncMock()
    return client


def _make_merged_entity(name: str, label: str) -> Entity:
    return Entity(
        name=name,
        label=label,
        provenance=Provenance(chunk_ids=[]),
    )


def _make_merged_relation(
    head: str,
    tail: str,
    rel_type: str,
) -> Relation:
    return Relation(
        head_name=head,
        tail_name=tail,
        type=rel_type,
        provenance=Provenance(chunk_ids=[]),
    )


class TestBuildExtractionPipeline:
    def test_returns_compiled_graph(self, tmp_path):
        cfg = _no_extractor_config(tmp_path)
        app = build_extraction_pipeline(cfg)
        assert app is not None

    def test_build_with_custom_schema(self, tmp_path):
        mini_schema = GraphSchema(
            entity_types=[
                EntityType(
                    label="Drug",
                    gliner_label="drug",
                    description="d",
                    key_property="name",
                ),
            ],
            relation_types=[],
        )
        cfg = _no_extractor_config(tmp_path)
        app = build_extraction_pipeline(cfg, schema=mini_schema)
        assert app is not None

    def test_build_without_checkpointer(self, tmp_path):
        cfg = _no_extractor_config(tmp_path)
        app = build_extraction_pipeline(cfg, checkpointer=None)
        assert app is not None

    def test_build_with_neo4j_client(self, tmp_path):
        cfg = _no_extractor_config(tmp_path)
        app = build_extraction_pipeline(cfg, neo4j_client=_mock_neo4j_client())
        assert app is not None


class TestPrepare:
    async def test_generates_batch_id_when_missing(self):
        result = await prepare({"batch_id": "", "chunks": []})
        assert result["batch_id"] != ""

    async def test_generates_batch_id_when_absent(self):
        result = await prepare({"chunks": []})
        assert result["batch_id"] != ""

    async def test_keeps_existing_batch_id(self):
        result = await prepare({"batch_id": "my-batch", "chunks": []})
        assert result["batch_id"] == "my-batch"


class TestResolveStep:
    """Exercises the `resolve` closure's report-building + routing logic."""

    async def test_no_entities_short_circuits_to_end(self, tmp_path):
        """When no entities in state, _has_entities routes to END (no build_report)."""
        cfg = _no_extractor_config(tmp_path)
        app = build_extraction_pipeline(cfg)
        chunks = [
            Document(external_id="c1", text="No entities here.", source="textbooks"),
        ]

        result = await app.ainvoke({"chunks": chunks, "batch_id": "empty-batch"})

        assert "extraction_report" not in result
        assert "ingestion_report" not in result

    async def test_report_built_from_state_when_entities_present(
        self,
        tmp_path,
    ):
        """When entities are in state, the closure builds a full ExtractionReport."""
        cfg = _no_extractor_config(tmp_path)
        entities = [_make_merged_entity("metformin", "Drug")]

        app = build_extraction_pipeline(cfg)
        result = await app.ainvoke(
            {
                "entities": entities,
                "chunks": [],
                "batch_id": "report-batch",
            }
        )

        report = result["extraction_report"]
        assert isinstance(report, ExtractionReport)
        assert report.batch_id == "report-batch"
        assert report.chunk_count == 0
        assert report.entities_after_resolution == 1
        assert result["output_path"] == str(tmp_path / "report-batch")

    async def test_routes_to_end_without_neo4j_client(self, tmp_path):
        """With entities but no neo4j_client, only extraction_report is set."""
        cfg = _no_extractor_config(tmp_path)
        entities = [_make_merged_entity("metformin", "Drug")]

        app = build_extraction_pipeline(cfg)
        result = await app.ainvoke(
            {
                "entities": entities,
                "chunks": [],
                "batch_id": "no-neo4j",
            }
        )

        assert "ingestion_report" not in result
        assert result["extraction_report"] is not None

    async def test_routes_to_ingest_to_graph_db_with_neo4j_client(
        self,
        tmp_path,
    ):
        """Extraction and ingestion reports set w/ entities + neo4j_client."""
        cfg = _no_extractor_config(tmp_path)
        entities = [_make_merged_entity("metformin", "Drug")]
        relations = [
            _make_merged_relation("metformin", "t2dm", "TREATED_BY"),
        ]

        mock_client = _mock_neo4j_client()
        mock_client.write_knowledge_graph = AsyncMock()

        app = build_extraction_pipeline(cfg, neo4j_client=mock_client)
        result = await app.ainvoke(
            {
                "entities": entities,
                "relations": relations,
                "chunks": [],
                "batch_id": "with-neo4j",
            }
        )

        report = result["ingestion_report"]
        assert isinstance(report, IngestionReport)
        assert report.batch_id == "with-neo4j"
        assert report.entities_ingested == len(entities)
        assert report.entities_ingested == len(entities)
        mock_client.write_knowledge_graph.assert_awaited_once()


class TestIngestToGraphDbStep:
    async def test_not_reached_when_resolve_routes_to_end(self, tmp_path):
        """No entities extracted -> resolve routes to END -> ingest node not reached."""
        cfg = _no_extractor_config(tmp_path)
        mock_client = _mock_neo4j_client()

        app = build_extraction_pipeline(cfg, neo4j_client=mock_client)
        chunks = [
            Document(
                external_id="c1",
                text="Metformin treats T2DM.",
                source="textbooks",
            ),
        ]
        result = await app.ainvoke({"chunks": chunks, "batch_id": "test"})

        assert "ingestion_report" not in result
        mock_client.ingest_entities.assert_not_called()
        mock_client.ingest_relations.assert_not_called()

    async def test_absent_without_neo4j_client(self, tmp_path):
        cfg = _no_extractor_config(tmp_path)
        app = build_extraction_pipeline(cfg)
        chunks = [
            Document(
                external_id="c1",
                text="Metformin treats T2DM.",
                source="textbooks",
            ),
        ]
        result = await app.ainvoke({"chunks": chunks, "batch_id": "test"})
        assert "ingestion_report" not in result
