"""Build structured-text context for community LLM report generation."""

from __future__ import annotations

from am_diag.common.data_models.community import Community
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation


def build_community_context(
    community: Community,
    all_entities: dict[str, Entity],
    all_relations: list[Relation],
    max_input_length: int = 8000,
    sub_reports: dict[str, str] | None = None,
) -> str:
    """Build structured-text context for a community's LLM report.

    Uses graph-based context: entities as bullet points and relationships
    as directed statements. Substitutes sub-community reports at higher
    levels when available, respecting ``max_input_length`` token budget.

    Args:
        community: Target community.
        all_entities: All entities keyed by UUID string.
        all_relations: All relations.
        max_input_length: Max chars for the output context.
        sub_reports: Mapping of ``community_id → report_summary`` for
            child communities.

    Returns:
        Context string truncated to ``max_input_length``.
    """
    lines: list[str] = []
    entity_id_strs = {str(uid) for uid in community.entity_ids}

    lines.append(f"Entities ({len(entity_id_strs)}):")
    for uid in community.entity_ids:
        ent = all_entities.get(str(uid))
        if ent:
            lines.append(f"- {ent.name} [{ent.label}]")

    community_relations = [
        rel
        for rel in all_relations
        if str(rel.head_id) in entity_id_strs and str(rel.tail_id) in entity_id_strs
    ]
    lines.append(f"\nRelationships ({len(community_relations)}):")
    for rel in community_relations:
        head_str = str(rel.head_id) if rel.head_id else ""
        tail_str = str(rel.tail_id) if rel.tail_id else ""
        head_name = (
            all_entities[head_str].name if head_str in all_entities else head_str
        )
        tail_name = (
            all_entities[tail_str].name if tail_str in all_entities else tail_str
        )
        lines.append(f"- {head_name} --{rel.type}--> {tail_name}")

    if sub_reports:
        child_reports = {
            cid: summary
            for cid, summary in sub_reports.items()
            if cid.startswith(f"{community.level + 1}:")
        }
        if child_reports:
            lines.append("\nSub-community summaries:")
            for cid, summary in child_reports.items():
                lines.append(f"[{cid}]: {summary[:500]}")

    context = "\n".join(lines)
    return context[:max_input_length]


__all__ = ["build_community_context"]
