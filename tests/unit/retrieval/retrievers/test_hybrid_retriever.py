"""Tests for the HybridRetriever template and its concrete retrievers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from am_diag.common.data_models.chunk import Chunk
from am_diag.common.data_models.entity import Entity
from am_diag.common.data_models.vector_record import VectorHit
from am_diag.retrieval.config import RetrievalConfig
from am_diag.retrieval.retrievers.chunk import ChunkRetriever
from am_diag.retrieval.retrievers.community import CommunityRetriever
from am_diag.retrieval.retrievers.entity import EntityRetriever
from am_diag.retrieval.retrievers.relation import RelationRetriever


_HYBRID = "am_diag.retrieval.retrievers.base.hybrid_search"
_BFS = "am_diag.retrieval.retrievers.base.bfs_expand"


def _entity(graph: AsyncMock | None = None) -> EntityRetriever:
    return EntityRetriever(
        config=RetrievalConfig(),
        vector_store=AsyncMock(),
        graph_store=graph or AsyncMock(),
        embedder=AsyncMock(),
    )


def _community(graph: AsyncMock | None = None) -> CommunityRetriever:
    return CommunityRetriever(
        config=RetrievalConfig(),
        vector_store=AsyncMock(),
        graph_store=graph or AsyncMock(),
        embedder=AsyncMock(),
    )


class TestHybridRetriever:
    """Tests for hybrid retriever."""

    async def test_empty_hits_returns_empty(self) -> None:
        """Empty hits returns empty."""
        retriever = _entity()
        with patch(_HYBRID, AsyncMock(return_value=[])):
            assert await retriever.retrieve("q") == []

    async def test_hydrate_path_builds_full_entity(self) -> None:
        """Hydrate path builds full entity."""
        eid = uuid4()
        hit = VectorHit(id=eid, score=0.8, payload={"name": "payload-name"})
        graph = AsyncMock()
        graph.execute_query = AsyncMock(
            return_value=[
                {
                    "id": str(eid),
                    "props": {
                        "name": "metformin",
                        "label": "Drug",
                        "description": "an antidiabetic",
                        "score": 0.7,
                        "canonical_name": "metformin",
                        "aliases": ["glucophage"],
                    },
                },
            ],
        )
        retriever = _entity(graph)
        with patch(_HYBRID, AsyncMock(return_value=[hit])):
            results = await retriever.retrieve("metformin")

        assert len(results) == 1
        entity = results[0].item
        assert isinstance(entity, Entity)
        assert entity.name == "metformin"
        assert entity.description == "an antidiabetic"
        assert entity.aliases == ["glucophage"]
        assert results[0].source == "entity"
        assert graph.execute_query.await_count == 1  # hydration query

    async def test_payload_path_skips_hydration(self) -> None:
        """Payload path skips hydration."""
        hit = VectorHit(
            id=uuid4(),
            score=0.5,
            payload={"text": "hello", "document_id": str(uuid4())},
        )
        graph = AsyncMock()
        retriever = ChunkRetriever(
            config=RetrievalConfig(),
            vector_store=AsyncMock(),
            graph_store=graph,
            embedder=AsyncMock(),
        )
        with patch(_HYBRID, AsyncMock(return_value=[hit])):
            results = await retriever.retrieve("q")

        assert isinstance(results[0].item, Chunk)
        assert results[0].item.text == "hello"
        graph.execute_query.assert_not_called()  # no hydration on payload path

    async def test_bfs_expands_from_seeds(self) -> None:
        """Bfs expands from seeds."""
        eid = uuid4()
        hit = VectorHit(id=eid, score=0.9, payload={"name": "x", "label": "Drug"})
        retriever = _entity()
        retriever._graph.execute_query = AsyncMock(return_value=[])  # empty hydrate
        with (
            patch(_HYBRID, AsyncMock(return_value=[hit])),
            patch(_BFS, AsyncMock(return_value=[])) as mock_bfs,
        ):
            await retriever.retrieve("q", bfs=True)

        mock_bfs.assert_awaited_once()
        assert mock_bfs.await_args.args[1] == [str(eid)]  # seeds = the entity hits

    async def test_bfs_skipped_when_no_seeds(self) -> None:
        """Bfs skipped when no seeds."""
        hit = VectorHit(
            id=uuid4(), score=0.6, payload={"title": "t", "community_id": str(uuid4())}
        )
        graph = AsyncMock()
        graph.execute_query = AsyncMock(return_value=[])  # seed query finds nothing
        retriever = _community(graph)
        with (
            patch(_HYBRID, AsyncMock(return_value=[hit])),
            patch(_BFS, AsyncMock(return_value=[])) as mock_bfs,
        ):
            await retriever.retrieve("q", bfs=True)

        mock_bfs.assert_not_awaited()

    async def test_relation_seed_ids_from_endpoints(self) -> None:
        """Relation seed ids from endpoints."""
        retriever = RelationRetriever(
            config=RetrievalConfig(),
            vector_store=AsyncMock(),
            graph_store=AsyncMock(),
            embedder=AsyncMock(),
        )
        rid, head, tail = uuid4(), uuid4(), uuid4()
        rows = {
            str(rid): {
                "id": str(rid),
                "props": {},
                "head_id": str(head),
                "tail_id": str(tail),
            },
        }
        seeds = await retriever._seed_ids([VectorHit(id=rid, score=1.0)], rows)
        assert seeds == [str(head), str(tail)]

    async def test_config_resolved_by_name(self) -> None:
        """Config resolved by name."""
        config = RetrievalConfig()
        retriever = _entity()
        assert retriever._collection == config.entity_collection
        assert retriever._top_k == config.entity_top_k
        # community has no score_threshold field -> getattr default None
        assert _community()._score_threshold is None

    async def test_retrieve_all_at_level_uses_reports_cypher(self) -> None:
        """Retrieve all at level uses reports cypher."""
        graph = AsyncMock()
        rid = uuid4()
        graph.execute_query = AsyncMock(
            return_value=[
                {
                    "id": str(rid),
                    "community_id": str(uuid4()),
                    "title": "T",
                    "summary": "S",
                }
            ],
        )
        results = await _community(graph).retrieve_all_at_level(level=1)

        assert len(results) == 1
        assert results[0].source == "community_all"
        graph.execute_query.assert_awaited_once()
        query, params = graph.execute_query.await_args.args
        assert "HAS_REPORT" in query
        assert params == {"level": 1}
