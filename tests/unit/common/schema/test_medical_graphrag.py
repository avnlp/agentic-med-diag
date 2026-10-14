"""Tests for MEDICAL_GRAPHRAG_SCHEMA — entity types, relation types, and labels."""

from am_diag.common.schema.medical_graphrag import (
    ENTITY_TYPES,
    MEDICAL_GRAPHRAG_SCHEMA,
    RELATION_TYPES,
)


class TestMedicalGraphRAGSchema:
    """Tests for the MEDICAL_GRAPHRAG_SCHEMA constant."""

    def test_gliner_labels_count(self):
        """Verify count of GLiNER labels."""
        assert len(MEDICAL_GRAPHRAG_SCHEMA.gliner_labels()) == 13

    def test_gliner_to_label_drug(self):
        """Verify GLiNER label maps to Drug entity type."""
        assert (
            MEDICAL_GRAPHRAG_SCHEMA.gliner_to_label[
                "drug, medication, or pharmaceutical agent"
            ]
            == "Drug"
        )

    def test_glirel_labels_count(self):
        """Verify count of GLiREL relation labels."""
        assert len(MEDICAL_GRAPHRAG_SCHEMA.glirel_labels()) == 25

    def test_entity_prompt_block_contains_all_types(self):
        """Verify entity prompt block includes all expected types."""
        block = MEDICAL_GRAPHRAG_SCHEMA.entity_prompt_block()
        expected = [
            "Disease",
            "Drug",
            "DrugClass",
            "Symptom",
            "Pathogen",
            "AnatomicalStructure",
            "Procedure",
            "DiagnosticTest",
            "RiskFactor",
            "Gene",
            "Protein",
            "Pathway",
            "MechanismOfAction",
        ]
        for name in expected:
            assert name in block

    def test_relation_prompt_block_contains_all_types(self):
        """Verify relation prompt block includes all expected types."""
        block = MEDICAL_GRAPHRAG_SCHEMA.relation_prompt_block()
        expected = [
            "HAS_SYMPTOM",
            "TREATED_BY",
            "DIAGNOSED_BY",
            "CAUSED_BY",
            "HAS_GENETIC_CAUSE",
            "AFFECTS",
            "HAS_COMPLICATION",
            "DIFFERENTIAL_FOR",
            "HAS_RISK_FACTOR",
            "IS_A",
            "BELONGS_TO_CLASS",
            "TARGETS",
            "INHIBITS",
            "ACTIVATES",
            "METABOLIZED_BY",
            "INTERACTS_WITH",
            "CONTRAINDICATED_IN",
            "CAUSES_ADVERSE_EFFECT",
            "MONITORED_BY",
            "ENCODES",
            "PARTICIPATES_IN",
            "PART_OF",
            "INNERVATED_BY",
            "SUPPLIED_BY",
            "HAS_MECHANISM",
        ]
        for name in expected:
            assert name in block

    def test_all_relation_head_tail_labels_valid(self):
        """Verify all relation head/tail labels reference valid entity types."""
        valid_entity_labels = {et.label for et in ENTITY_TYPES}
        for rt in RELATION_TYPES:
            for label in rt.head_labels:
                assert label in valid_entity_labels, (
                    f"{rt.type} head_label '{label}' not in ENTITY_TYPES"
                )
            for label in rt.tail_labels:
                assert label in valid_entity_labels, (
                    f"{rt.type} tail_label '{label}' not in ENTITY_TYPES"
                )
