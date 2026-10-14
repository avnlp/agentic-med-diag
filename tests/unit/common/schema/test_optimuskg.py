"""Tests for the generated OptimusKG union schema.

The schema is derived mechanically from the abbreviation map and the 27 structural
edge labels; these assertions pin the label set, the endpoint derivation, and the
structural-placeholder fallback used until semantic relations are enumerated.
"""

from __future__ import annotations

from am_diag.common.schema.optimuskg import (
    ABBREV_TO_LABEL,
    OPTIMUSKG_SCHEMA,
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

    def test_placeholder_relation_per_structural_label(self) -> None:
        """With no enumerated semantics, each structural label yields one type."""
        assert len(OPTIMUSKG_SCHEMA.relation_types) == len(STRUCTURAL_LABELS)
        types = {rt.type for rt in OPTIMUSKG_SCHEMA.relation_types}
        assert "ANA_GEN" in types
