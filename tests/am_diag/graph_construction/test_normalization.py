"""Tests for am_diag/graph_construction/normalize.py."""

from __future__ import annotations

from am_diag.common.data_models import Entity, Relation
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.graph_construction import EntityRelationNormalizer


def _entity(
    name: str,
    label: str,
    score: float = 1.0,
    extractor: ExtractorName = ExtractorName.GLINER,
    chunk_id: str = "c1",
) -> Entity:
    return Entity(
        name=name,
        label=label,
        score=score,
        provenance=Provenance(
            extractor=extractor,
            chunk_ids=[],
        ),
    )


def _relation(
    head: str,
    tail: str,
    rel_type: str,
    score: float = 1.0,
    extractor: ExtractorName = ExtractorName.LLM,
    chunk_id: str = "c1",
) -> Relation:
    return Relation(
        head_name=head,
        tail_name=tail,
        type=rel_type,
        score=score,
        provenance=Provenance(
            extractor=extractor,
            chunk_ids=[],
        ),
    )


class TestEntityRelationNormalizer:
    async def test_normalize_deduplicates_entities(self):
        """Same entity from different extractors merged."""
        normalizer = EntityRelationNormalizer.__new__(EntityRelationNormalizer)
        normalizer._config = type(
            "cfg",
            (),
            {"min_entity_score": 0.0, "min_entity_length": 0},
        )()

        entities = [
            _entity("Metformin", "Drug", extractor=ExtractorName.GLINER),
            _entity("metformin", "Drug", extractor=ExtractorName.LLM),
        ]
        norm_entities, norm_relations = await normalizer.normalize(entities, [], [])
        assert len(norm_entities) == 1
        assert norm_entities[0].name in ("Metformin", "metformin")
        assert ExtractorName.GLINER in norm_entities[0].provenance.sources
        assert ExtractorName.LLM in norm_entities[0].provenance.sources

    async def test_normalize_empty_input(self):
        normalizer = EntityRelationNormalizer.__new__(EntityRelationNormalizer)
        normalizer._config = type(
            "cfg",
            (),
            {"min_entity_score": 0.0, "min_entity_length": 0},
        )()

        norm_entities, norm_relations = await normalizer.normalize([], [], [])
        assert norm_entities == []
        assert norm_relations == []

    async def test_normalize_relations_endpoint_validation(self):
        """Relations with dangling endpoints are dropped."""
        normalizer = EntityRelationNormalizer.__new__(EntityRelationNormalizer)
        normalizer._config = type(
            "cfg",
            (),
            {"min_entity_score": 0.0, "min_entity_length": 0},
        )()

        entities = [_entity("metformin", "Drug")]
        relations = [
            _relation("type 2 diabetes", "metformin", "TREATED_BY"),
        ]
        _, norm_relations = await normalizer.normalize(entities, relations, [])
        assert len(norm_relations) == 0  # "type 2 diabetes" not in entities

    async def test_normalize_entity_not_found_provided(self):
        """Every chunk gets entity and relation lists even with zero raw extractions."""
        normalizer = EntityRelationNormalizer.__new__(EntityRelationNormalizer)
        normalizer._config = type(
            "cfg",
            (),
            {"min_entity_score": 0.0, "min_entity_length": 0},
        )()

        entities = [_entity("metformin", "Drug"), _entity("type 2 diabetes", "Disease")]
        relations = [
            _relation("type 2 diabetes", "metformin", "TREATED_BY"),
        ]
        norm_entities, norm_relations = await normalizer.normalize(
            entities,
            relations,
            [],
        )
        assert len(norm_entities) == 2
        assert len(norm_relations) == 1
