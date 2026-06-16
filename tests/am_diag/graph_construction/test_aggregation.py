"""Tests for am_diag/graph_construction/aggregate.py."""

from __future__ import annotations

from am_diag.common.data_models import Entity, Relation
from am_diag.common.data_models.provenance import Provenance
from am_diag.graph_construction import KGAggregator


def _entity(
    name: str,
    label: str,
    score: float = 1.0,
    sources: list | None = None,
    surface_forms: list[str] | None = None,
    chunk_ids: list | None = None,
) -> Entity:
    return Entity(
        name=name,
        label=label,
        score=score,
        provenance=Provenance(
            sources=sources or [],
            surface_forms=surface_forms or [name],
            chunk_ids=chunk_ids or [],
        ),
    )


def _relation(
    head: str,
    tail: str,
    rel_type: str,
    score: float = 1.0,
    sources: list | None = None,
    chunk_ids: list | None = None,
) -> Relation:
    return Relation(
        head_name=head,
        tail_name=tail,
        type=rel_type,
        score=score,
        provenance=Provenance(
            sources=sources or [],
            chunk_ids=chunk_ids or [],
        ),
    )


class TestKGAggregator:
    async def test_aggregates_entities_across_chunks(self):
        agg = KGAggregator()
        entities = [
            _entity("Metformin", "Drug", chunk_ids=[]),
            _entity("metformin", "Drug", chunk_ids=[]),
            _entity("Lisinopril", "Drug", chunk_ids=[]),
        ]
        agg_entities, _ = await agg.aggregate(entities, [])
        # metformin and lisinopril: metformin collapsed to 1, lisinopril is distinct
        assert len(agg_entities) == 2

    async def test_merges_surface_forms_on_collision(self):
        agg = KGAggregator()
        entities = [
            _entity("Metformin", "Drug", surface_forms=["Metformin"]),
            _entity("metformin", "Drug", surface_forms=["metformin"]),
        ]
        agg_entities, _ = await agg.aggregate(entities, [])
        assert len(agg_entities) == 1
        assert "Metformin" in agg_entities[0].provenance.surface_forms
        assert "metformin" in agg_entities[0].provenance.surface_forms

    async def test_empty_input(self):
        agg = KGAggregator()
        agg_entities, agg_relations = await agg.aggregate([], [])
        assert agg_entities == []
        assert agg_relations == []

    async def test_aggregates_to_lowercase(self):
        agg = KGAggregator()
        entities = [_entity("Metformin", "Drug")]
        agg_entities, _ = await agg.aggregate(entities, [])
        assert agg_entities[0].name == "metformin"
