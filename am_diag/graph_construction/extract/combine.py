"""Combine multiple extractor results (intersection/union)."""

from __future__ import annotations

from typing import Literal

from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.relation import Relation
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.resolve.text import normalize_text


Strategy = Literal["union", "intersection", "max_score", "gliner_primary"]


def _entity_key(e: Entity) -> tuple[str, str]:
    return (normalize_text(e.name), e.label)


def _relation_key(r: Relation) -> tuple[str, str, str]:
    return (normalize_text(r.head_name), normalize_text(r.tail_name), r.type)


def _merge_entity(into: Entity, src: Entity) -> None:
    for cid in src.provenance.chunk_ids:
        if cid not in into.provenance.chunk_ids:
            into.provenance.chunk_ids.append(cid)
    for s in src.provenance.surface_forms:
        if s not in into.provenance.surface_forms:
            into.provenance.surface_forms.append(s)
    into.score = max(into.score, src.score)
    into.schema_properties.update(src.schema_properties)


def _merge_relation(into: Relation, src: Relation) -> None:
    for cid in src.provenance.chunk_ids:
        if cid not in into.provenance.chunk_ids:
            into.provenance.chunk_ids.append(cid)
    into.score = max(into.score, src.score)
    into.schema_properties.update(src.schema_properties)


def _union(
    results: list[tuple[list[Entity], list[Relation]]],
) -> tuple[list[Entity], list[Relation]]:
    seen_e: dict[tuple[str, str], Entity] = {}
    seen_r: dict[tuple[str, str, str], Relation] = {}
    for entities, relations in results:
        for e in entities:
            key = _entity_key(e)
            if key in seen_e:
                _merge_entity(seen_e[key], e)
            else:
                seen_e[key] = e
        for r in relations:
            key = _relation_key(r)
            if key in seen_r:
                _merge_relation(seen_r[key], r)
            else:
                seen_r[key] = r
    return list(seen_e.values()), list(seen_r.values())


def _set_intersection(
    results: list[tuple[list[Entity], list[Relation]]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    if not results:
        return set(), set()
    e_sets = [{_entity_key(e) for e in entities} for entities, _ in results]
    r_sets = [{_relation_key(r) for r in relations} for _, relations in results]
    common_e = e_sets[0]
    for s in e_sets[1:]:
        common_e &= s
    common_r = r_sets[0]
    for s in r_sets[1:]:
        common_r &= s
    return common_e, common_r


def _filter_by_keys(
    results: list[tuple[list[Entity], list[Relation]]],
    keep_e: set[tuple[str, str]],
    keep_r: set[tuple[str, str, str]],
) -> tuple[list[Entity], list[Relation]]:
    all_e: dict[tuple[str, str], Entity] = {}
    all_r: dict[tuple[str, str, str], Relation] = {}
    for entities, relations in results:
        for e in entities:
            key = _entity_key(e)
            if key in keep_e:
                if key in all_e:
                    _merge_entity(all_e[key], e)
                else:
                    all_e[key] = e
        for r in relations:
            key = _relation_key(r)
            if key in keep_r:
                if key in all_r:
                    _merge_relation(all_r[key], r)
                else:
                    all_r[key] = r
    return list(all_e.values()), list(all_r.values())


def _intersection(
    results: list[tuple[list[Entity], list[Relation]]],
) -> tuple[list[Entity], list[Relation]]:
    common_e, common_r = _set_intersection(results)
    return _filter_by_keys(results, common_e, common_r)


def _max_score(
    results: list[tuple[list[Entity], list[Relation]]],
) -> tuple[list[Entity], list[Relation]]:
    seen_e: dict[tuple[str, str], Entity] = {}
    seen_r: dict[tuple[str, str, str], Relation] = {}
    for entities, relations in results:
        for e in entities:
            key = _entity_key(e)
            if key not in seen_e or e.score > seen_e[key].score:
                seen_e[key] = e
        for r in relations:
            key = _relation_key(r)
            if key not in seen_r or r.score > seen_r[key].score:
                seen_r[key] = r
    return list(seen_e.values()), list(seen_r.values())


def _gliner_primary(
    results: list[tuple[list[Entity], list[Relation]]],
) -> tuple[list[Entity], list[Relation]]:
    gliner_idx = -1
    for i, (entities, _) in enumerate(results):
        if entities and any(
            e.provenance.extractor == ExtractorName.GLINER for e in entities
        ):
            gliner_idx = i
            break
    if gliner_idx < 0:
        return _union(results)
    base_e, base_r = results[gliner_idx]
    base_e_map: dict[tuple[str, str], Entity] = {_entity_key(e): e for e in base_e}
    base_r_map: dict[tuple[str, str, str], Relation] = {
        _relation_key(r): r for r in base_r
    }
    for entities, relations in results:
        if (entities, relations) == (base_e, base_r):
            continue
        for e in entities:
            key = _entity_key(e)
            if key not in base_e_map:
                base_e_map[key] = e
        for r in relations:
            key = _relation_key(r)
            if key not in base_r_map:
                base_r_map[key] = r
    return list(base_e_map.values()), list(base_r_map.values())


_STRATEGIES: dict[str, Strategy] = {
    "union": "union",
    "intersection": "intersection",
    "max_score": "max_score",
    "gliner_primary": "gliner_primary",
}


class ExtractionCombiner:
    """Merge extraction results from multiple extractors using a strategy.

    Strategies:
    - ``union`` — superset across all extractors, merge provenance.
    - ``intersection`` — only items present in every extractor.
    - ``max_score`` — keep the highest-scoring version per key.
    - ``gliner_primary`` — start with GLiNER output, supplement with LLM.
    """

    def __init__(self, config: ExtractionPipelineConfig) -> None:
        """Initialize combiner with pipeline config."""
        self._strategy: Strategy = config.merge_strategy

    def combine(
        self,
        results: list[tuple[list[Entity], list[Relation]]],
    ) -> tuple[list[Entity], list[Relation]]:
        """Merge multiple extraction results per the configured strategy.

        Args:
            results: ``(entities, relations)`` from each extractor.

        Returns:
            Merged ``(entities, relations)``.
        """
        if not results:
            return [], []
        if len(results) == 1:
            return results[0]
        fn = {
            "union": _union,
            "intersection": _intersection,
            "max_score": _max_score,
            "gliner_primary": _gliner_primary,
        }[self._strategy]
        return fn(results)


__all__ = ["ExtractionCombiner"]
