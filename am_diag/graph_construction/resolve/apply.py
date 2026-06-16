"""Apply alias maps to entities and relations."""

from __future__ import annotations

from am_diag.common.data_models.cluster import ResolvedAliasMap
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation


class AliasApplier:
    """Apply alias maps to entities and relations.

    Sets ``canonical_name`` / ``aliases`` on entities and
    ``head_id`` / ``tail_id`` on relations.
    """

    def apply(
        self,
        entities: list[Entity],
        relations: list[Relation],
        entity_alias_map: ResolvedAliasMap,
        edge_alias_map: ResolvedAliasMap | None = None,  # noqa: ARG002
    ) -> tuple[list[Entity], list[Relation]]:
        """Apply entity alias resolution to entities and relations.

        Mutates DataPoints in place:
        - Entities get ``canonical_name`` and ``aliases``.
        - Relations get ``head_id`` / ``tail_id``.

        Args:
            entities: Aggregated entities.
            relations: Aggregated relations.
            entity_alias_map: Canonical → members mapping.
            edge_alias_map: Unused (reserved for edge-type resolution).

        Returns:
            ``(merged_entities, merged_relations)``.
        """
        text_to_entity = {e.name: e for e in entities}
        merged_entities: list[Entity] = []
        text_to_merged: dict[str, Entity] = {}

        for alias, members in entity_alias_map.alias_to_members.items():
            matched = [text_to_entity[m] for m in members if m in text_to_entity]
            if not matched:
                continue
            best = max(matched, key=lambda e: e.score)
            all_chunk_ids: list = []
            all_props: dict = {}
            all_surface_forms: set[str] = set()
            for e in matched:
                for cid in e.provenance.chunk_ids:
                    if cid not in all_chunk_ids:
                        all_chunk_ids.append(cid)
                all_props.update(e.schema_properties)
                all_surface_forms.update(e.provenance.surface_forms)
            all_surface_forms.update(m for m in members if m)

            best.canonical_name = alias
            best.aliases = sorted(all_surface_forms)
            best.provenance.chunk_ids = all_chunk_ids
            best.schema_properties = all_props
            merged_entities.append(best)

            for member in members:
                text_to_merged[member] = best
            text_to_merged[alias] = best
            for e in matched:
                text_to_merged[e.name] = best
                for sf in e.provenance.surface_forms:
                    text_to_merged.setdefault(sf, best)

        seen_rels: set[tuple[str, str, str]] = set()
        merged_relations: list[Relation] = []
        for rel in relations:
            head_merged = text_to_merged.get(rel.head_name)
            tail_merged = text_to_merged.get(rel.tail_name)
            if head_merged is None or tail_merged is None:
                continue
            key = (str(head_merged.id), str(tail_merged.id), rel.type)
            if key in seen_rels:
                continue
            seen_rels.add(key)
            rel.head_id = head_merged.id
            rel.tail_id = tail_merged.id
            merged_relations.append(rel)

        return merged_entities, merged_relations


__all__ = ["AliasApplier"]
