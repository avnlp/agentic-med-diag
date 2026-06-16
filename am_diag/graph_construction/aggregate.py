"""Component: cross-chunk aggregation of normalised extraction results."""

from __future__ import annotations

from am_diag.common.data_models import Entity, Relation
from am_diag.graph_construction.resolve.text import normalize_text


class KGAggregator:
    r"""Component — combines per-chunk graphs into one lowercase-normalised graph.

    Pure set-union aggregation across all chunk extractions:
    unions entities (keyed by `(lowercased text, label)`) and
    relation triples — merging `sources`/`surface_forms`/`chunk_ids`/
    `schema_properties` on collision and keeping the highest-scoring surface
    form as canonical text.

    All free-text entity names and relation endpoint strings are lowercased
    (NFKC + whitespace-collapse + lowercase, via `normalize_text`).
    Schema-fixed identifiers (`label`, `type`) are never modified.
    Requires no LLM.
    """

    async def aggregate(  # noqa: PLR0912
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> tuple[list[Entity], list[Relation]]:
        """Cross-chunk union of entities/relations.

        Args:
            entities: Normalized-stage entities per chunk.
            relations: Normalized-stage relations per chunk.

        Returns:
            `(aggregated_entities, aggregated_relations)`.
        """
        # ── Aggregate entities ──────────────────────────────────────────────
        entity_accum: dict[tuple[str, str], Entity] = {}
        for ent in entities:
            lower_text = normalize_text(ent.name)
            key = (lower_text, ent.label)
            if key in entity_accum:
                existing = entity_accum[key]
                for s in ent.provenance.sources:
                    if s not in existing.provenance.sources:
                        existing.provenance.sources.append(s)
                for sf in ent.provenance.surface_forms:
                    if sf not in existing.provenance.surface_forms:
                        existing.provenance.surface_forms.append(sf)
                for cid in ent.provenance.chunk_ids:
                    if cid not in existing.provenance.chunk_ids:
                        existing.provenance.chunk_ids.append(cid)
                if ent.score > existing.score:
                    existing.score = ent.score
                    existing.name = ent.name
                existing.schema_properties.update(ent.schema_properties)
            else:
                ent.name = lower_text
                entity_accum[key] = ent

        aggregated_entities = list(entity_accum.values())

        # ── Build entity lookup for relation endpoint validation ───────────
        lower_to_canonical: dict[str, str] = {}
        for ent in aggregated_entities:
            lower_to_canonical[ent.name] = ent.name
            for sf in ent.provenance.surface_forms:
                lower_to_canonical.setdefault(normalize_text(sf), ent.name)

        # ── Aggregate relations ─────────────────────────────────────────────
        rel_accum: dict[tuple[str, str, str], Relation] = {}
        for rel in relations:
            lower_head = normalize_text(rel.head_name)
            lower_tail = normalize_text(rel.tail_name)
            if (
                lower_head not in lower_to_canonical
                or lower_tail not in lower_to_canonical
            ):
                continue
            key = (lower_head, lower_tail, rel.type)
            if key in rel_accum:
                existing = rel_accum[key]
                for s in rel.provenance.sources:
                    if s not in existing.provenance.sources:
                        existing.provenance.sources.append(s)
                for cid in rel.provenance.chunk_ids:
                    if cid not in existing.provenance.chunk_ids:
                        existing.provenance.chunk_ids.append(cid)
                existing.score = max(existing.score, rel.score)
                existing.schema_properties.update(rel.schema_properties)
            else:
                rel.head_name = lower_head
                rel.tail_name = lower_tail
                rel_accum[key] = rel

        aggregated_relations = list(rel_accum.values())

        return aggregated_entities, aggregated_relations


__all__ = ["KGAggregator"]
