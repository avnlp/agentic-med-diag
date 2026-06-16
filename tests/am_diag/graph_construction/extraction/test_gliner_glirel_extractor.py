"""Tests for GLiNER/GLiREL extractor internal helpers."""

from __future__ import annotations

from am_diag.common.data_models import Entity
from am_diag.common.data_models.provenance import Provenance
from am_diag.graph_construction.extract.glirel_relations import _entities_for_glirel


class TestEntitiesForGlirel:
    def test_converts_entity_to_glirel_format(self):
        """Single entity with char offsets produces correct GLiREL token indices."""
        entity = Entity(
            name="Metformin",
            label="Drug",
            provenance=Provenance(
                start_char=0,
                end_char=9,
            ),
        )
        # token positions: M-e-t-f-o-r-m-i-n(0-9) treats(10-16) diabetes(17-25)
        char_to_token = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        result = _entities_for_glirel([entity], char_to_token)
        assert len(result) == 1
        assert result[0] == [0, 0, "Drug", "Metformin"]

    def test_skips_entity_without_char_offsets(self):
        """Entities without start_char/end_char are skipped."""
        entity = Entity(
            name="Metformin",
            label="Drug",
            provenance=Provenance(),
        )
        result = _entities_for_glirel([entity], {})
        assert result == []

    def test_handles_multiple_entities(self):
        """Multiple entities are all converted."""
        entity1 = Entity(
            name="Metformin",
            label="Drug",
            provenance=Provenance(
                start_char=0,
                end_char=9,
            ),
        )
        entity2 = Entity(
            name="diabetes",
            label="Disease",
            provenance=Provenance(
                start_char=17,
                end_char=25,
            ),
        )
        entity3 = Entity(
            name="insulin",
            label="Drug",
            provenance=Provenance(
                start_char=32,
                end_char=38,
            ),
        )
        char_to_token = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            17: 2,
            18: 2,
            19: 2,
            20: 2,
            21: 2,
            22: 2,
            23: 2,
            24: 2,
            32: 4,
            33: 4,
            34: 4,
            35: 4,
            36: 4,
            37: 4,
        }
        result = _entities_for_glirel([entity1, entity2, entity3], char_to_token)
        assert len(result) == 3
