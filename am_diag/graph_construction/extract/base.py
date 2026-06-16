"""Abstract base class for entity/relation extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.relation import Relation


class Extractor(ABC):
    """Abstract base for all entity/relation extractors.

    Each extractor takes a list of ``Chunk`` data points and returns
    ``(entities, relations)`` with provenance tracking.
    """

    @abstractmethod
    async def extract(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[Entity], list[Relation]]:
        """Run extraction over *chunks*.

        Args:
            chunks: Document chunks to extract from.

        Returns:
            ``(entities, relations)`` extracted from the chunks.
        """


__all__ = ["Extractor"]
