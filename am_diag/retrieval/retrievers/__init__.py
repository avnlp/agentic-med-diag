"""Scope orchestrators — one public ``retrieve(query, filters)`` per retriever.

Each retriever wraps one or more atomic search methods and knows how to
hydrate raw ``VectorHit`` results into full ``SearchResult`` items.
"""

from __future__ import annotations

from am_diag.retrieval.retrievers.chunk import ChunkRetriever
from am_diag.retrieval.retrievers.community import CommunityRetriever
from am_diag.retrieval.retrievers.entity import EntityRetriever
from am_diag.retrieval.retrievers.relation import RelationRetriever
from am_diag.retrieval.retrievers.text2cypher import Text2CypherRetriever


__all__ = [
    "ChunkRetriever",
    "CommunityRetriever",
    "EntityRetriever",
    "RelationRetriever",
    "Text2CypherRetriever",
]
