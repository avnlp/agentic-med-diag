"""Named search recipes — data-only ``SearchConfig`` constants.

Adding a new retrieval strategy means defining a new constant here;
no orchestration code needs to change.
"""

from __future__ import annotations

from am_diag.retrieval.config import SearchConfig


# Single-method recipes
ENTITY: SearchConfig = SearchConfig(
    methods=["entity"],
    reranker="rrf",
    limit=20,
)

RELATION: SearchConfig = SearchConfig(
    methods=["relation"],
    reranker="rrf",
    limit=20,
)

CHUNK: SearchConfig = SearchConfig(
    methods=["chunk"],
    reranker="rrf",
    limit=20,
)

COMMUNITY: SearchConfig = SearchConfig(
    methods=["community"],
    reranker="rrf",
    limit=20,
)

# Multi-method hybrid recipes
HYBRID_RRF: SearchConfig = SearchConfig(
    methods=["entity", "relation", "chunk", "community"],
    reranker="rrf",
    bfs=False,
    limit=20,
)

HYBRID_CROSS_ENCODER: SearchConfig = SearchConfig(
    methods=["entity", "relation", "chunk", "community"],
    reranker="cross_encoder",
    bfs=True,
    limit=20,
)

# Specialised recipes
BFS_EXPAND: SearchConfig = SearchConfig(
    methods=["entity"],
    reranker="rrf",
    bfs=True,
    bfs_depth=2,
    limit=50,
)

TEXT2CYPHER: SearchConfig = SearchConfig(
    methods=["text2cypher"],
    reranker="rrf",
    limit=20,
)

# Convenience grouping for testing / iteration
ALL_RECIPES: list[tuple[str, SearchConfig]] = [
    ("entity", ENTITY),
    ("relation", RELATION),
    ("chunk", CHUNK),
    ("community", COMMUNITY),
    ("hybrid_rrf", HYBRID_RRF),
    ("hybrid_cross_encoder", HYBRID_CROSS_ENCODER),
    ("bfs_expand", BFS_EXPAND),
    ("text2cypher", TEXT2CYPHER),
]

__all__ = [
    "ALL_RECIPES",
    "BFS_EXPAND",
    "CHUNK",
    "COMMUNITY",
    "ENTITY",
    "HYBRID_CROSS_ENCODER",
    "HYBRID_RRF",
    "RELATION",
    "TEXT2CYPHER",
]
