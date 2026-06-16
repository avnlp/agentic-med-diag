"""Protocol interfaces for LLM completion and embedding generation.

These are base contracts used by BAML wrappers and downstream components.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Interface for LLM completion calls."""

    async def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a completion prompt to the LLM and return the response.

        Args:
            prompt: The user/assistant prompt text.
            system_prompt: Optional system-level instruction.

        Returns:
            The generated text response.
        """


@runtime_checkable
class Embedder(Protocol):
    """Interface for embedding generation."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: Texts to embed.

        Returns:
            One embedding vector per input text, in the same order as `texts`.
        """
