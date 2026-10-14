"""Shared fixtures for graph_construction tests."""

from __future__ import annotations

import pytest

from am_diag.common.data_models import Document, Entity, Relation
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.schema import MEDICAL_GRAPHRAG_SCHEMA
from am_diag.graph_construction.config import ExtractionPipelineConfig


@pytest.fixture
def sample_entities() -> list:
    """Sample Entity DataPoints for pipeline tests."""
    return [
        Entity(
            name="metformin",
            label="Drug",
            score=0.97,
            provenance=Provenance(
                extractor=ExtractorName.GLINER,
                chunk_ids=[],
            ),
        ),
        Entity(
            name="type 2 diabetes mellitus",
            label="Disease",
            score=0.95,
            provenance=Provenance(
                extractor=ExtractorName.LLM,
                chunk_ids=[],
            ),
        ),
    ]


@pytest.fixture
def sample_relations() -> list:
    """Sample Relation DataPoints for pipeline tests."""
    return [
        Relation(
            head_name="type 2 diabetes mellitus",
            tail_name="metformin",
            type="TREATED_BY",
            score=0.93,
            provenance=Provenance(
                extractor=ExtractorName.LLM,
                chunk_ids=[],
            ),
        ),
    ]


@pytest.fixture
def sample_merged_entities() -> list:
    """Sample merged Entity DataPoints for Neo4j record tests."""
    return [
        Entity(
            name="metformin",
            label="Drug",
            canonical_name="metformin",
            aliases=["Metformin"],
            score=0.97,
            provenance=Provenance(
                chunk_ids=[],
            ),
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Two sample document chunks for extraction tests."""
    return [
        Document(
            text=(
                "Metformin is a biguanide used as first-line therapy for type 2 diabetes mellitus. "
                "It inhibits mitochondrial complex I, reducing hepatic glucose production. "
                "Common adverse effects include nausea and diarrhea. "
                "Metformin is contraindicated in severe renal impairment (eGFR < 30 mL/min)."
            ),
            source="textbooks",
            external_id="tb_001",
            title="Pharmacology",
        ),
        Document(
            text=(
                "Type 2 diabetes mellitus (T2DM) is characterized by insulin resistance. "
                "HbA1c > 6.5% is diagnostic. First-line treatment includes metformin. "
                "Complications include diabetic nephropathy and retinopathy."
            ),
            source="statpearls",
            external_id="sp_042",
        ),
    ]


@pytest.fixture
def minimal_config(tmp_path: str) -> ExtractionPipelineConfig:
    """Minimal extraction config with all extractors disabled."""
    return ExtractionPipelineConfig(
        use_gliner=False,
        use_glirel=False,
        use_llm=False,
        output_dir=str(tmp_path),
    )


@pytest.fixture
def default_schema():
    """Default MEDICAL_GRAPHRAG_SCHEMA."""
    return MEDICAL_GRAPHRAG_SCHEMA


@pytest.fixture
def sample_chunk() -> Document:
    """Single sample document for extraction tests."""
    return Document(external_id="c1", text="Sample text.", source="textbooks")
