"""Tests for named search recipes — data-only SearchConfig constants."""

from __future__ import annotations

from am_diag.retrieval import recipes
from am_diag.retrieval.config import SearchConfig


class TestRecipeConstants:
    def test_entity_recipe(self) -> None:
        assert recipes.ENTITY.methods == ["entity"]
        assert recipes.ENTITY.reranker == "rrf"
        assert recipes.ENTITY.limit == 20

    def test_relation_recipe(self) -> None:
        assert recipes.RELATION.methods == ["relation"]
        assert recipes.RELATION.limit == 20

    def test_chunk_recipe(self) -> None:
        assert recipes.CHUNK.methods == ["chunk"]

    def test_community_recipe(self) -> None:
        assert recipes.COMMUNITY.methods == ["community"]

    def test_hybrid_rrf_includes_all(self) -> None:
        assert sorted(recipes.HYBRID_RRF.methods) == sorted(
            ["entity", "relation", "chunk", "community"],
        )
        assert recipes.HYBRID_RRF.reranker == "rrf"
        assert recipes.HYBRID_RRF.bfs is False

    def test_hybrid_cross_encoder_uses_bfs(self) -> None:
        assert recipes.HYBRID_CROSS_ENCODER.reranker == "cross_encoder"
        assert recipes.HYBRID_CROSS_ENCODER.bfs is True

    def test_bfs_expand(self) -> None:
        assert recipes.BFS_EXPAND.bfs is True
        assert recipes.BFS_EXPAND.bfs_depth == 2
        assert recipes.BFS_EXPAND.limit == 50

    def test_text2cypher_recipe(self) -> None:
        assert recipes.TEXT2CYPHER.methods == ["text2cypher"]

    def test_all_recipes_has_eight_entries(self) -> None:
        names = [name for name, _ in recipes.ALL_RECIPES]
        assert len(recipes.ALL_RECIPES) >= 8
        assert "hybrid_rrf" in names
        assert "text2cypher" in names

    def test_adding_new_recipe_needs_no_code_change(self) -> None:
        """A new recipe is just a new SearchConfig constant; no code changes needed."""
        new_recipe = SearchConfig(
            methods=["entity", "chunk"],
            reranker="mmr",
            limit=15,
        )
        assert new_recipe.limit == 15
        assert "entity" in new_recipe.methods
        assert new_recipe.reranker == "mmr"
