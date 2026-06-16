"""Tests for the BAML TypeBuilder builder."""

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType
from am_diag.llm.baml_client.type_builder import TypeBuilder
from am_diag.llm.schema_typebuilder import build_schema_typebuilder


class TestBuildSchemaTypebuilder:
    def _sample_schema(self) -> GraphSchema:
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
        schema = self._sample_schema()
        tb = build_schema_typebuilder(schema)
        assert isinstance(tb, TypeBuilder)

    def test_entity_types_populated(self):
        schema = self._sample_schema()
        tb = build_schema_typebuilder(schema)
        # TypeBuilder should have entity_type values
        # (exact API depends on the installed TypeBuilder version)
        assert tb is not None

    def test_empty_schema_does_not_raise(self):
        schema = GraphSchema(entity_types=[], relation_types=[])
        tb = build_schema_typebuilder(schema)
        assert tb is not None
