"""Consumer-owned role interfaces for the ingestion engine.

These Protocols are the seam that keeps the dependency direction inward: the
ingestion engine depends on the small interface it needs, and the higher-level
``api`` facade (``Graph``) satisfies it structurally. The engine never imports
``api``; the facade depends on the engines through these Protocols.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.schema import GraphSchema
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.vector.embedding.base import Embedder


class IngestionTarget(Protocol):
    """The live dependencies a job needs to write into one graph.

    ``api.Graph`` satisfies this structurally, so a job takes the ``Graph``
    itself with no adapter. The members are read-only.

    Attributes:
        graph_store: The graph store the job writes the knowledge graph to.
        vector_store: The vector store the job upserts embeddings to.
        embedder: The embedder; ``embedder.model_id`` keys the embedding cache.
        schema: The graph schema stamped onto written knowledge graphs and used
            to build the extraction pipeline.
        collections: Logical->physical vector collection names for this graph.
    """

    @property
    def graph_store(self) -> GraphStore:
        """The graph store the job writes the knowledge graph to."""
        ...

    @property
    def vector_store(self) -> VectorStoreBase:
        """The vector store the job upserts embeddings to."""
        ...

    @property
    def embedder(self) -> Embedder:
        """The embedder; ``embedder.model_id`` keys the embedding cache."""
        ...

    @property
    def schema(self) -> GraphSchema:
        """The schema stamped on written graphs and used for extraction."""
        ...

    @property
    def collections(self) -> Mapping[str, str]:
        """Logical->physical vector collection names for this graph."""
        ...


class Answerer(Protocol):
    """A project that answers one question over an optional graph subset.

    ``api.Project`` satisfies this structurally, so a ``QAJob`` depends only on
    the answering behaviour it needs and never imports ``api``. ``question`` is
    positional-only so an implementation may name its first parameter differently
    (``Project.answer`` accepts ``str | list[Message]``).
    """

    async def answer(
        self, question: str, /, *, graphs: list[str] | None = None
    ) -> ClinicalAnswer:
        """Answer ``question`` over an optional subset of member graphs."""
        ...
