"""BFS graph expansion from seed entity UUIDs.

Runs the ``retrieval/bfs_expand.cypher`` Cypher template over ``:RELATES_TO``
edges, degree-capped for medical supernodes.
"""

from __future__ import annotations

from typing import Any

from am_diag.common.cypher.loader import load
from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.retrieval.filters import SearchFilters


async def bfs_expand(
    graph_store: GraphStore,
    seed_ids: list[str],
    depth: int = 2,
    limit: int = 50,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """BFS expansion from ``seed_ids`` up to ``depth`` hops via ``:RELATES_TO``.

    Args:
        graph_store: Graph database client.
        seed_ids: Entity UUID strings to start traversal from.
        depth: Max hop count (default 2).
        limit: Max nodes to return (caps fan-out for medical supernodes).
        filters: Optional search filters.

    Returns:
        ``SearchResult`` items, one per neighbour node found.
    """
    if not seed_ids:
        return []

    cypher = load("retrieval/bfs_expand")
    params: dict[str, Any] = {
        "seed_ids": seed_ids,
        "depth": depth,
        "limit": limit,
    }
    where_clause, where_params = filters.to_cypher_where("n") if filters else ("", {})
    params.update(where_params)

    # Inject WHERE clause before RETURN
    if where_clause:
        cypher = cypher.replace("RETURN n.id", f"{where_clause}\nRETURN n.id")

    rows = await graph_store.execute_query(cypher, params)

    results: list[SearchResult] = []
    for row in rows:
        results.append(
            SearchResult(
                item={"id": row["id"], "name": row["name"], "labels": row["labels"]},
                score=1.0 / (1 + row["hop"]),
                source="bfs",
                matched_on=f"hop_{row['hop']}",
            ),
        )
    return results


__all__ = ["bfs_expand"]
