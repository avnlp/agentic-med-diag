"""Tests for GraphSchema (list-based API)."""

from __future__ import annotations

from am_diag.common.data_models.schema import EntityType, GraphSchema, RelationType


class TestGraphSchema:
    def test_custom_schema_gliner_labels(self):
        schema = GraphSchema(
            entity_types=[
                EntityType(
                    label="Disease",
                    gliner_label="disease or condition",
                    description="A disease.",
                    key_property="name",
                ),
                EntityType(
                    label="Drug",
                    gliner_label="drug or medication",
                    description="A drug.",
                    key_property="genericName",
                ),
            ],
            relation_types=[],
        )
        assert len(schema.gliner_labels()) == 2

    def test_gliner_to_label_mapping(self):
        schema = GraphSchema(
            entity_types=[
                EntityType(
                    label="Disease",
                    gliner_label="disease or condition",
                    description="A disease.",
                    key_property="name",
                ),
            ],
            relation_types=[],
        )
        assert schema.gliner_to_label["disease or condition"] == "Disease"

    def test_glirel_to_type_mapping(self):
        schema = GraphSchema(
            entity_types=[],
            relation_types=[
                RelationType(
                    type="TREATED_BY",
                    glirel_label="disease is treated by drug",
                    head_labels=["Disease"],
                    tail_labels=["Drug"],
                ),
            ],
        )
        assert schema.glirel_to_type["disease is treated by drug"] == "TREATED_BY"

    def test_entity_prompt_block_contains_description(self):
        schema = GraphSchema(
            entity_types=[
                EntityType(
                    label="Disease",
                    gliner_label="disease or condition",
                    description="A named disease or condition.",
                    key_property="name",
                ),
            ],
            relation_types=[],
        )
        block = schema.entity_prompt_block()
        assert "Disease" in block
        assert "A named disease or condition." in block

    def test_relation_prompt_block_contains_head_and_tail(self):
        schema = GraphSchema(
            entity_types=[],
            relation_types=[
                RelationType(
                    type="TREATED_BY",
                    glirel_label="disease is treated by drug",
                    head_labels=["Disease"],
                    tail_labels=["Drug", "Procedure"],
                ),
            ],
        )
        block = schema.relation_prompt_block()
        assert "TREATED_BY" in block
        assert "Disease" in block
        assert "Drug" in block
        assert "Procedure" in block


class TestEntityType:
    def test_default_key_property(self):
        entity_type = EntityType(
            label="Disease",
            gliner_label="disease or condition",
            description="A disease.",
        )
        assert entity_type.key_property == "name"

    def test_uuid_is_deterministic(self):
        et1 = EntityType(
            label="Drug",
            gliner_label="drug or medication",
            description="A drug.",
        )
        et2 = EntityType(
            label="Drug",
            gliner_label="drug or medication",
            description="A drug.",
        )
        assert et1.id == et2.id


class TestRelationType:
    def test_properties_default_to_empty_list(self):
        relation_type = RelationType(
            type="TREATED_BY",
            glirel_label="disease is treated by drug",
            head_labels=["Disease"],
            tail_labels=["Drug"],
        )
        assert relation_type.properties == []

    def test_uuid_is_deterministic(self):
        rt1 = RelationType(
            type="TREATED_BY",
            glirel_label="disease is treated by drug",
            head_labels=["Disease"],
            tail_labels=["Drug"],
        )
        rt2 = RelationType(
            type="TREATED_BY",
            glirel_label="disease is treated by drug",
            head_labels=["Disease"],
            tail_labels=["Drug"],
        )
        assert rt1.id == rt2.id
