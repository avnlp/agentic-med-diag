"""LLM-based community summarization (bottom-up report generation)."""

from __future__ import annotations

import asyncio
import logging

from am_diag.common.data_models.community import Community
from am_diag.common.data_models.community_report import CommunityReport, Finding
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.enums import ExtractorName
from am_diag.common.data_models.provenance import Provenance
from am_diag.common.data_models.relation import Relation
from am_diag.common.data_models.schema import GraphSchema
from am_diag.graph_construction.community.context import build_community_context


logger = logging.getLogger(__name__)


async def _generate_report(
    community: Community,
    context_text: str,
    schema: GraphSchema,
    max_report_length: int,
) -> CommunityReport | None:
    from am_diag.llm.baml_client import b  # noqa: PLC0415

    try:
        schema_context = (
            f"{schema.entity_prompt_block()}\n{schema.relation_prompt_block()}"
        )
        result = await b.GenerateCommunityReport(
            context_text,
            schema_context,
            max_report_length,
        )
        return CommunityReport(
            community_id=community.id,
            report_type="full",
            title=result.title,
            summary=result.summary,
            findings=[
                Finding(summary=f.summary, explanation=f.explanation)
                for f in result.findings
            ],
            rating=result.rating,
            rating_explanation=result.rating_explanation,
            rank=result.rating,
            provenance=Provenance(
                sources=[ExtractorName.COMMUNITY_REPORT],
            ),
        )
    except Exception:
        logger.exception(
            "Failed to generate report for community %s",
            community.community_id,
        )
        return None


class CommunitySummarizer:
    """Generate LLM community reports bottom-up.

    Pure component — takes communities + entities + relations,
    returns ``list[CommunityReport]`` DataPoints.
    """

    def __init__(
        self,
        max_input_length: int = 8000,
        max_report_length: int = 2000,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize summarizer with concurrency and length limits."""
        self._max_input_length = max_input_length
        self._max_report_length = max_report_length
        self._max_concurrent = max_concurrent

    async def summarize(
        self,
        communities: list[Community],
        entities: list[Entity],
        relations: list[Relation],
        schema: GraphSchema,
    ) -> list[CommunityReport]:
        """Generate reports for all communities, bottom-up by level.

        Args:
            communities: ``Community`` DataPoints from the detector.
            entities: All merged ``Entity`` DataPoints.
            relations: All merged ``Relation`` DataPoints.
            schema: KG schema for prompt context.

        Returns:
            List of ``CommunityReport`` DataPoints (failed reports omitted).
        """
        if not communities:
            return []

        entity_map = {str(e.id): e for e in entities}

        communities_by_level = sorted(
            communities,
            key=lambda c: c.level,
        )

        semaphore = asyncio.Semaphore(self._max_concurrent)
        reports: list[CommunityReport] = []
        sub_report_summaries: dict[str, str] = {}

        for community in communities_by_level:
            context = build_community_context(
                community,
                entity_map,
                relations,
                self._max_input_length,
                sub_reports=sub_report_summaries,
            )

            async def _limited(c: Community, ctx: str) -> CommunityReport | None:
                async with semaphore:
                    return await _generate_report(
                        c,
                        ctx,
                        schema,
                        self._max_report_length,
                    )

            report = await _limited(community, context)
            if report is not None:
                reports.append(report)
                sub_report_summaries[community.community_id] = report.summary

        return reports


__all__ = ["CommunitySummarizer"]
