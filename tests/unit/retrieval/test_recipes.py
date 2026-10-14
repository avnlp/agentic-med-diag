"""Tests for named search recipes — data-only SearchConfig constants."""

from __future__ import annotations

from am_diag.retrieval import recipes
from am_diag.retrieval.config import SearchConfig


class TestRecipeConstants:
    """Tests for the named search recipe constants (SearchConfig instances)."""

    def test_entity_recipe(self) -> None:
        """Verify ENTITY recipe uses entity method with RRF reranker."""
        assert recipes.ENTITY.methods == ["entity"]
        assert recipes.ENTITY.reranker == "rrf"
        assert recipes.ENTITY.limit == 20

    def test_relation_recipe(self) -> None:
        """Verify RELATION recipe uses relation method with default limit."""
        assert recipes.RELATION.methods == ["relation"]
        assert recipes.RELATION.limit == 20

    def test_chunk_recipe(self) -> None:
        """Verify CHUNK recipe uses chunk method."""
        assert recipes.CHUNK.methods == ["chunk"]

    def test_community_recipe(self) -> None:
        """Verify COMMUNITY recipe uses community method."""
        assert recipes.COMMUNITY.methods == ["community"]

    def test_hybrid_rrf_includes_all(self) -> None:
        """Verify HYBRID_RRF recipe includes all four search methods."""
        assert sorted(recipes.HYBRID_RRF.methods) == sorted(
            ["entity", "relation", "chunk", "community"],
        )
        assert recipes.HYBRID_RRF.reranker == "rrf"
        assert recipes.HYBRID_RRF.bfs is False

    def test_hybrid_cross_encoder_uses_bfs(self) -> None:
        """Verify HYBRID_CROSS_ENCODER recipe uses cross_encoder with BFS enabled."""
        assert recipes.HYBRID_CROSS_ENCODER.reranker == "cross_encoder"
        assert recipes.HYBRID_CROSS_ENCODER.bfs is True

    def test_bfs_expand(self) -> None:
        """Verify BFS_EXPAND recipe enables BFS with depth 2 and limit 50."""
        assert recipes.BFS_EXPAND.bfs is True
        assert recipes.BFS_EXPAND.bfs_depth == 2
        assert recipes.BFS_EXPAND.limit == 50

    def test_text2cypher_recipe(self) -> None:
        """Verify TEXT2CYPHER recipe uses text2cypher method."""
        assert recipes.TEXT2CYPHER.methods == ["text2cypher"]

    def test_all_recipes_has_eight_entries(self) -> None:
        """Verify ALL_RECIPES contains at least eight named recipes."""
        names = [name for name, _ in recipes.ALL_RECIPES]
        assert len(recipes.ALL_RECIPES) >= 8
        assert "hybrid_rrf" in names
        assert "text2cypher" in names

    def test_adding_new_recipe_needs_no_code_change(self) -> None:
        """Verify a new recipe can be added as a SearchConfig constant."""
        new_recipe = SearchConfig(
            methods=["entity", "chunk"],
            reranker="mmr",
            limit=15,
        )
        assert new_recipe.limit == 15
        assert "entity" in new_recipe.methods
        assert new_recipe.reranker == "mmr"
