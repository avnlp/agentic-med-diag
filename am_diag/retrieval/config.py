"""Retrieval pipeline configuration — env-overridable with RETRIEVAL_ prefix.

Collapsed from 3 relation collections to a single ``relation`` collection.
``SearchConfig`` is a data-only recipe type that selects which retriever
methods to run and how to fuse their results.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class RetrievalConfig(BaseSettings):
    """All fields env-overridable with RETRIEVAL_ prefix.

    Four vector collections (entity, relation, chunk, community) replace the
    previous 3-relation-collection scheme.
    """

    model_config = {"env_prefix": "RETRIEVAL_", "env_file": ".env", "extra": "ignore"}

    # Collection names — must match the embedding pipeline config
    entity_collection: str = "am_diag_entity"
    relation_collection: str = "am_diag_relation"
    chunk_collection: str = "am_diag_chunk"
    community_collection: str = "am_diag_community"

    # Entity search
    entity_top_k: int = 10
    entity_score_threshold: float | None = None

    # Relation search
    relation_top_k: int = 10
    relation_score_threshold: float | None = None

    # Chunk search
    chunk_top_k: int = 10
    chunk_score_threshold: float | None = None

    # Community report search
    community_top_k: int = 5
    community_report_level: int = 0

    # k-hop BFS traversal
    traversal_depth: int = 2
    traversal_limit: int = 50

    # Text-to-Cypher
    use_text_to_cypher: bool = True
    text2cypher_max_attempts: int = 2

    # Hybrid RRF
    rrf_k: int = 60
    hybrid_alpha: float = 0.5

    # Reranker
    reranker_min_score: float = 0.0


class SearchConfig(BaseModel):
    """Data-only recipe: which methods to run, which reranker, and result caps.

    A new retrieval strategy is simply a new constant of this type — no code
    changes are needed.
    """

    methods: list[Literal["entity", "relation", "chunk", "community", "text2cypher"]]
    reranker: Literal["rrf", "cross_encoder", "mmr", "node_distance"] = "rrf"
    bfs: bool = False
    bfs_depth: int = 2
    limit: int = 20
    filters: dict[str, Any] | None = None


__all__ = ["RetrievalConfig", "SearchConfig"]
