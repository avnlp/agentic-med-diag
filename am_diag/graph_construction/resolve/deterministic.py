"""No-LLM deterministic deduplication (exact-normalize, singularize, SemHash)."""

from __future__ import annotations

from semhash import SemHash

from am_diag.common.data_models.cluster import ResolvedAliasMap
from am_diag.graph_construction.config import ExtractionPipelineConfig
from am_diag.graph_construction.resolve.text import normalize_text, singularize


class DeterministicDeduplicator:
    """No-LLM deduplication: exact-normalize, singularize, and SemHash.

    Intended as a pre-filter before LLM-based resolution to reduce
    candidate-set size.
    """

    def __init__(self, config: ExtractionPipelineConfig) -> None:
        """Initialize deduplicator with pipeline config."""
        self._config = config

    def dedupe(
        self,
        items: list[str],
        kind: str = "entity",
    ) -> ResolvedAliasMap:
        """Deduplicate *items* using deterministic string matching.

        Steps:
        1. ``normalize_text`` merge (exact after NFKC + lowercase).
        2. Optional ``singularize`` merge.
        3. SemHash ``self_deduplicate`` at ``dedupe_semhash_threshold``.

        Args:
            items: Strings to deduplicate.
            kind: Entity kind label (for logging / SemHash namespace).

        Returns:
            Mapping of canonical alias → members.
        """
        use_lsh = self._config.use_deterministic_dedupe

        alias_to_members: dict[str, list[str]] = {}

        for item in items:
            norm = normalize_text(item)
            sing = singularize(norm) if use_lsh else norm

            key = sing

            found = False
            for alias, members in alias_to_members.items():
                if normalize_text(alias) == key:
                    if item not in members:
                        members.append(item)
                    found = True
                    break

            if not found:
                alias_to_members[key] = [item]

        if use_lsh:
            alias_to_members = self._semhash_dedupe(
                alias_to_members,
                self._config.dedupe_semhash_threshold,
            )

        return ResolvedAliasMap(alias_to_members=alias_to_members)

    @staticmethod
    def _semhash_dedupe(
        alias_to_members: dict[str, list[str]],
        threshold: float,
    ) -> dict[str, list[str]]:
        keys = list(alias_to_members.keys())
        semhash = SemHash.from_records(records=keys)
        result = semhash.self_deduplicate(threshold=threshold)

        deduped: dict[str, list[str]] = {}
        for item in result.selected_with_duplicates:
            canonical = item.record
            members = list(alias_to_members[canonical])
            for dup_record, _score in item.duplicates:
                members.extend(alias_to_members[dup_record])
            deduped[canonical] = members

        # Remaining keys not in any dedup group
        processed = {item.record for item in result.selected_with_duplicates}
        for key in keys:
            if key not in processed:
                deduped[key] = list(alias_to_members[key])

        return deduped


__all__ = ["DeterministicDeduplicator"]
