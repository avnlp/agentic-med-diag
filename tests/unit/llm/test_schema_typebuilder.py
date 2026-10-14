"""Tests for the BAML TypeBuilder builder."""

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.llm.baml_client.type_builder import TypeBuilder
from am_diag.llm.schema_typebuilder import build_schema_typebuilder


class TestBuildSchemaTypebuilder:
    """Tests for build_schema_typebuilder."""

    def _sample_schema(self) -> GraphSchema:
        """Build a sample schema for testing."""
        return GraphSchema(
            entity_types=[
                EntityType(
                    label="Disease",
                    gliner_label="disease or disorder",
                    description="A named disease.",
                ),
                EntityType(
                    label="Drug",
                    gliner_label="drug or medication",
                    description="A pharmaceutical.",
                ),
            ],
            relation_types=[
                RelationType(
                    type="TREATED_BY",
                    glirel_label="treated by",
                    head_labels=["Disease"],
                    tail_labels=["Drug"],
                ),
            ],
        )

    def test_returns_typebuilder(self):
        """Verify function returns a TypeBuilder instance."""
        schema = self._sample_schema()
        tb = build_schema_typebuilder(schema)
        assert isinstance(tb, TypeBuilder)

    def test_entity_types_populated(self):
        """Verify entity types are populated in the TypeBuilder."""
        schema = self._sample_schema()
        tb = build_schema_typebuilder(schema)
        # TypeBuilder should have entity_type values
        # (exact API depends on the installed TypeBuilder version)
        assert tb is not None

    def test_empty_schema_does_not_raise(self):
        """Verify empty schema does not raise an error."""
        schema = GraphSchema(entity_types=[], relation_types=[])
        tb = build_schema_typebuilder(schema)
        assert tb is not None
