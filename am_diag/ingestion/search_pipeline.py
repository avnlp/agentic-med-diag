"""Search pipeline — LangGraph StateGraph wrapping ``SearchEngine``.

Provides a graph interface for the multi-strategy search, used by the
agentic query harness and eval framework.
"""

from __future__ import annotations

from typing import Any, TypedDict

from am_diag.common.data_models.search import SearchResult
from am_diag.db.graph.base import GraphStore
from am_diag.db.vector.base import VectorStoreBase
from am_diag.retrieval import recipes
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.search import SearchEngine
from am_diag.vector.embedding.base import Embedder
from am_diag.vector.reranking.base import Reranker


class SearchPipelineState(TypedDict, total=False):
    """State for the search pipeline graph."""

    question: str
    recipe: str
    search_result: list[SearchResult]


def build_search_pipeline(
    config: RetrievalConfig | None = None,
    vector_store: VectorStoreBase | None = None,
    graph_store: GraphStore | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a compiled LangGraph StateGraph for search.

    Args:
        config: Retrieval configuration.
        vector_store: Vector store.
        graph_store: Graph database client.
        embedder: Embedder.
        reranker: Optional cross-encoder.
        checkpointer: Optional LangGraph checkpointer.

    Returns:
        A compiled ``CompiledGraph``.
    """
    from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415
    from langgraph.graph import END, START, StateGraph  # noqa: PLC0415

    engine = SearchEngine(
        config or RetrievalConfig(),
        vector_store,  # type: ignore
        graph_store,  # type: ignore
        embedder,  # type: ignore
        reranker=reranker,
    )

    async def _search(state: SearchPipelineState) -> dict:
        question = state.get("question", "")
        recipe_name = state.get("recipe", "hybrid_rrf")
        recipe = getattr(recipes, recipe_name.upper(), None)
        if recipe is None:
            results = await engine.search(question, recipe_name)
        else:
            results = await engine.search(question, recipe)
        return {"search_result": results}

    builder: StateGraph = StateGraph(SearchPipelineState)  # type: ignore
    builder.add_node("search", _search)
    builder.add_edge(START, "search")
    builder.add_edge("search", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


__all__ = ["SearchPipelineState", "build_search_pipeline"]
