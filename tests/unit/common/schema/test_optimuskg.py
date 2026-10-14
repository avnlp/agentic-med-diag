"""Tests for the generated OptimusKG union schema.

The schema is derived mechanically from the abbreviation map, the 27 structural
edge labels, and the semantic relation values enumerated from the released data;
these assertions pin the label set, the endpoint derivation, and the relation types.
"""

from __future__ import annotations

from am_diag.common.schema.optimuskg import (
    ABBREV_TO_LABEL,
    OPTIMUSKG_SCHEMA,
    SEMANTIC_RELATIONS,
    STRUCTURAL_LABELS,
    _endpoints,
)


class TestOptimusKGSchema:
    """Entity-label coverage, endpoint mapping, and relation generation."""

    def test_has_ten_entity_types(self) -> None:
        """One EntityType per unique PascalCase label (10 OptimusKG node types)."""
        assert len(OPTIMUSKG_SCHEMA.entity_types) == 10

    def test_entity_labels_are_the_pascalcase_set(self) -> None:
        """Entity labels are exactly the abbreviation map's target labels."""
        labels = {et.label for et in OPTIMUSKG_SCHEMA.entity_types}
        assert labels == set(ABBREV_TO_LABEL.values())

    def test_entity_identity_keyed_on_curie(self) -> None:
        """Structured entity types resolve by CURIE, not name."""
        gene = next(et for et in OPTIMUSKG_SCHEMA.entity_types if et.label == "Gene")
        assert gene.key_property == "curie"

    def test_endpoints_map_structural_label_to_pascalcase(self) -> None:
        """A HEAD-TAIL label resolves to its PascalCase endpoint labels."""
        assert _endpoints("ANA-GEN") == ("AnatomicalStructure", "Gene")
        assert _endpoints("DRG-DIS") == ("Drug", "Disease")

    def test_relation_endpoints_are_populated(self) -> None:
        """Every relation type carries head/tail labels from its structural pair."""
        for rt in OPTIMUSKG_SCHEMA.relation_types:
            assert rt.head_labels and rt.tail_labels

    def test_every_structural_label_has_semantic_relations(self) -> None:
        """No structural label falls back to the placeholder relation type."""
        assert set(SEMANTIC_RELATIONS) == set(STRUCTURAL_LABELS)
        assert all(SEMANTIC_RELATIONS[label] for label in STRUCTURAL_LABELS)

    def test_relation_types_are_the_released_semantic_values(self) -> None:
        """Relation types carry real values, each scoped to its endpoint labels."""
        by_type = {
            (rt.type, rt.head_labels[0], rt.tail_labels[0])
            for rt in OPTIMUSKG_SCHEMA.relation_types
        }
        assert ("EXPRESSION_PRESENT", "AnatomicalStructure", "Gene") in by_type
        assert ("INDICATION", "Drug", "Disease") in by_type
        assert ("CONTRAINDICATION", "Drug", "Disease") in by_type
        assert ("INHIBITOR", "Drug", "Gene") in by_type
        assert "ANA_GEN" not in {rt.type for rt in OPTIMUSKG_SCHEMA.relation_types}
