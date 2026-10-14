"""Composed extractor — GLiNER + GLiREL + LLM, combined, normalized, linked."""

from __future__ import annotations

import logging
from uuid import UUID

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.graph_construction._provenance import ensure_provenance
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.extract.base import Extractor
from am_diag.graph_construction.extract.combine import ExtractionCombiner
from am_diag.graph_construction.extract.gliner_entities import GLiNEREntityExtractor
from am_diag.graph_construction.extract.glirel_relations import GLiRELRelationExtractor
from am_diag.graph_construction.extract.llm import LLMExtractor
from am_diag.graph_construction.normalize import EntityRelationNormalizer
from am_diag.graph_construction.resolve.text import normalize_text


logger = logging.getLogger(__name__)


class ComposedExtractor(Extractor):
    """Run the configured extractors, then combine, normalize, and link edges.

    Produces entities with deterministic ``(name, label)`` ids and relations with
    ``head_id``/``tail_id`` set by name lookup so they survive the write layer
    (``neo4j.py`` drops relations lacking endpoint ids). GLiREL is driven via
    ``extract_from_entities`` with GLiNER's spans grouped per chunk, since its
    ``extract`` returns nothing on its own.
    """

    def __init__(self, config: ExtractionPipelineConfig, schema: GraphSchema) -> None:
        """Build the selected extractors, combiner, and normalizer."""
        self._config = config
        self._schema = schema
        self._gliner = (
            GLiNEREntityExtractor(config, schema) if config.use_gliner else None
        )
        self._glirel = (
            GLiRELRelationExtractor(config, schema) if config.use_glirel else None
        )
        self._llm = LLMExtractor(config, schema) if config.use_llm else None
        self._combiner = ExtractionCombiner(config)
        self._normalizer = EntityRelationNormalizer(config)

    async def extract(self, chunks: list[Chunk]) -> tuple[list[Entity], list[Relation]]:
        """Extract, combine, normalize, and endpoint-link over *chunks*."""
        results: list[tuple[list[Entity], list[Relation]]] = []

        gliner_entities: list[Entity] = []
        if self._gliner is not None:
            gliner_entities, _ = await self._gliner.extract(chunks)
            results.append((gliner_entities, []))

        if self._glirel is not None:
            per_chunk = self._group_by_chunk(gliner_entities, chunks)
            glirel_relations = await self._glirel.extract_from_entities(
                chunks, per_chunk
            )
            results.append(([], glirel_relations))

        if self._llm is not None:
            llm_entities, llm_relations = await self._llm.extract(chunks)
            results.append((llm_entities, llm_relations))

        entities, relations = self._combiner.combine(results)
        entities, relations = await self._normalizer.normalize(entities, relations, [])
        relations = self._link_endpoints(entities, relations)
        return entities, relations

    @staticmethod
    def _group_by_chunk(
        entities: list[Entity], chunks: list[Chunk]
    ) -> list[list[Entity]]:
        """Group GLiNER entities per chunk, aligned to the *chunks* order."""
        by_chunk: dict[UUID, list[Entity]] = {c.id: [] for c in chunks}
        for ent in entities:
            for cid in ensure_provenance(ent).chunk_ids:
                if cid in by_chunk:
                    by_chunk[cid].append(ent)
        return [by_chunk[c.id] for c in chunks]

    @staticmethod
    def _link_endpoints(
        entities: list[Entity], relations: list[Relation]
    ) -> list[Relation]:
        """Set ``head_id``/``tail_id`` by name and rebuild for a stable id.

        Relations whose endpoints are not in the entity set are dropped. Each
        kept relation is reconstructed (excluding ``id``) so that, once
        ``Relation`` marks ``(head_id, tail_id, type)`` as Dedup keys, the id
        becomes the deterministic ``uuid5`` derived from the now-set endpoints.
        """
        name_to_id = {normalize_text(e.name): e.id for e in entities}
        linked: list[Relation] = []
        dropped = 0
        for rel in relations:
            head_id = name_to_id.get(normalize_text(rel.head_name))
            tail_id = name_to_id.get(normalize_text(rel.tail_name))
            if head_id is None or tail_id is None:
                dropped += 1
                continue
            data = rel.model_dump(exclude={"id"})
            data["head_id"] = head_id
            data["tail_id"] = tail_id
            linked.append(Relation(**data))
        if dropped:
            logger.debug(
                "endpoint-link dropped %d/%d relations (unresolved endpoints)",
                dropped,
                len(relations),
            )
        return linked


__all__ = ["ComposedExtractor"]
