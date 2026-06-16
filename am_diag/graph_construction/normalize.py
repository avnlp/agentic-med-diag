"""Component: merge and normalize per-chunk extraction results."""

from __future__ import annotations

from am_diag.common.data_models import Document, Entity, Relation
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.resolve.text import normalize_text as _normalize


class EntityRelationNormalizer:
    """Component — merges all extractor results and normalises per chunk."""

    def __init__(self, config: ExtractionPipelineConfig) -> None:
        """Initialize normalizer with extraction pipeline config."""
        self._config = config

    async def normalize(  # noqa: PLR0912, PLR0915
        self,
        entities: list[Entity],
        relations: list[Relation],
        chunks: list[Document],
    ) -> tuple[list[Entity], list[Relation]]:
        """Normalize RAW entities/relations.

        Deduplicates by `(normalized name, label)` for entities and
        `(normalized head, normalized tail, type)` for relations.
        Merges sources, surface_forms, chunk_ids, score, and schema_properties
        on dedup collision.

        Args:
            entities: RAW-stage entities from all extractors.
            relations: RAW-stage relations from all extractors.
            chunks: Document chunks (for chunk_id order and threshold config).

        Returns:
            `(normalized_entities, normalized_relations)`.
        """
        cfg = self._config
        min_score = cfg.min_entity_score
        min_length = cfg.min_entity_length

        # ── Deduplicate entities: key=(normalized_text, label) ─────
        entity_accum: dict[tuple[str, str], Entity] = {}
        for ent in entities:
            if ent.score < min_score:
                continue
            norm_text = _normalize(ent.name)
            if len(norm_text) < min_length:
                continue
            key = (norm_text, ent.label)
            if key in entity_accum:
                existing = entity_accum[key]
                if (
                    ent.provenance.extractor
                    and ent.provenance.extractor not in existing.provenance.sources
                ):
                    existing.provenance.sources.append(ent.provenance.extractor)
                if ent.name not in existing.provenance.surface_forms:
                    existing.provenance.surface_forms.append(ent.name)
                for cid in ent.provenance.chunk_ids:
                    if cid not in existing.provenance.chunk_ids:
                        existing.provenance.chunk_ids.append(cid)
                if ent.score > existing.score:
                    existing.score = ent.score
                    existing.name = ent.name
                existing.schema_properties.update(ent.schema_properties)
            else:
                ent.provenance.sources = (
                    [ent.provenance.extractor] if ent.provenance.extractor else []
                )
                ent.provenance.surface_forms = [ent.name]
                entity_accum[key] = ent

        normalized_entities = list(entity_accum.values())

        # ── Build entity lookup for relation endpoint resolution ─────────
        norm_to_canonical: dict[str, str] = {}
        for ent in normalized_entities:
            norm_text = _normalize(ent.name)
            norm_to_canonical[norm_text] = ent.name
            for form in ent.provenance.surface_forms:
                norm_to_canonical.setdefault(_normalize(form), ent.name)

        # ── Deduplicate relations ────────────────────────────────────────
        accum_rels: dict[tuple[str, str, str], Relation] = {}
        for rel in relations:
            canon_head = norm_to_canonical.get(_normalize(rel.head_name))
            canon_tail = norm_to_canonical.get(_normalize(rel.tail_name))
            if canon_head is None or canon_tail is None:
                continue
            key = (_normalize(canon_head), _normalize(canon_tail), rel.type)
            if key in accum_rels:
                existing = accum_rels[key]
                for cid in rel.provenance.chunk_ids:
                    if cid not in existing.provenance.chunk_ids:
                        existing.provenance.chunk_ids.append(cid)
                existing.score = max(existing.score, rel.score)
            else:
                rel.head_name = canon_head
                rel.tail_name = canon_tail
                rel.provenance.chunk_ids = list(rel.provenance.chunk_ids)
                accum_rels[key] = rel

        normalized_relations: list[Relation] = []
        for rel in accum_rels.values():
            rel.provenance.sources = (
                [rel.provenance.extractor] if rel.provenance.extractor else []
            )
            normalized_relations.append(rel)

        return normalized_entities, normalized_relations


__all__ = ["EntityRelationNormalizer"]
