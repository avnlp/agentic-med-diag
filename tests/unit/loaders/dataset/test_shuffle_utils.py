"""Unit tests for MCQ option shuffling logic.

Covers shuffle_options() — anchoring, label-ref preservation, multi-answer,
determinism — plus regex patterns ANCHOR and LABEL_REF.
"""

from __future__ import annotations

import pytest

from am_diag.loaders.dataset.shuffle_utils import (
    ANCHOR,
    LABEL_REF,
    _stable_options_hash,
    shuffle_options,
)


class TestShuffleOptions:
    """Tests for shuffle_options(), ANCHOR, and LABEL_REF regex patterns."""

    def test_dict_input_returns_dict_str_answer(self) -> None:
        """Verify dict input preserves structure and returns a str answer key."""
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answer, _ = shuffle_options(options, "B", seed=0)
        assert isinstance(result, dict)
        assert isinstance(answer, str)
        assert answer in options

    def test_dict_input_preserves_answer_text(self) -> None:
        """Verify correct answer text survives dict-input shuffling."""
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answer, _ = shuffle_options(options, "B", seed=0)
        assert result[answer] == "Banana"

    def test_list_input_returns_list_str_answer(self) -> None:
        """Verify list input returns options list and string answer."""
        options = ["Apple", "Banana", "Cherry"]
        labels = ["A", "B", "C"]
        result, answer, _ = shuffle_options(options, "B", labels=labels, seed=0)
        assert isinstance(result, list)
        assert isinstance(answer, str)

    def test_list_input_without_labels_raises(self) -> None:
        """Verify list input without labels raises ValueError."""
        options = ["A", "B", "C"]
        with pytest.raises(ValueError, match="labels must be provided"):
            shuffle_options(options, "B", seed=0)

    def test_list_input_wrong_label_count_raises(self) -> None:
        """Verify mismatched label count raises ValueError."""
        options = ["A", "B", "C"]
        with pytest.raises(ValueError, match="labels length"):
            shuffle_options(options, "B", labels=["A", "B"], seed=0)

    def test_list_input_preserves_answer_text(self) -> None:
        """Verify correct answer text survives list-input shuffling."""
        options = ["Apple", "Banana", "Cherry"]
        labels = ["A", "B", "C"]
        result, answer, _ = shuffle_options(options, "B", labels=labels, seed=0)
        idx = labels.index(answer)
        assert result[idx] == "Banana"

    def test_anchors_in_list_input(self) -> None:
        """Verify anchor options stay in place in shuffled list input."""
        options = ["Option X", "All of the above", "Option Z"]
        labels = ["A", "B", "C"]
        result, _, _ = shuffle_options(options, "A", labels=labels, seed=42)
        assert result[1] == "All of the above"

    def test_multi_answer_dict_returns_list(self) -> None:
        """Verify multi-answer dict returns list of answer keys."""
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answers, _ = shuffle_options(options, ["A", "C"], seed=0)
        assert isinstance(result, dict)
        assert isinstance(answers, list)
        assert len(answers) == 2

    def test_multi_answer_all_labels_valid(self) -> None:
        """Verify all multi-answer keys present after shuffle."""
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answers, _ = shuffle_options(options, ["A", "C"], seed=0)
        assert all(a in result for a in answers)

    def test_multi_answer_list_options(self) -> None:
        """Verify multi-answer list returns lists for result and answers."""
        options = ["Apple", "Banana", "Cherry"]
        labels = ["A", "B", "C"]
        result, answers, _ = shuffle_options(options, ["A", "C"], labels=labels, seed=0)
        assert isinstance(result, list)
        assert isinstance(answers, list)
        assert len(answers) == 2

    def test_anchors_preserved_in_place(self) -> None:
        """Verify anchor options keep their original positions."""
        options = {
            "A": "Option A",
            "B": "All of the above",
            "C": "Option C",
            "D": "None of the above",
        }
        result, _, _ = shuffle_options(options, "A", seed=42)
        keys = list(result.keys())
        assert keys[1] == "B"
        assert keys[3] == "D"

    def test_anchor_block_building_edge(self) -> None:
        """Verify blocks.append fires with mid-list anchor and trailing tail."""
        options = [
            "All of the above",
            "Option B",
            "Option C",
            "None of the above",
            "Option E",
        ]
        labels = ["A", "B", "C", "D", "E"]
        result, answer, _ = shuffle_options(
            options,
            "C",
            labels=labels,
            seed=42,
        )
        assert result[0] == "All of the above"
        assert result[3] == "None of the above"
        assert set(result) == set(options)

    def test_label_refs_skip_shuffle(self) -> None:
        """Verify label-ref options like 'Both A and B' stay."""
        options = {
            "A": "Option A",
            "B": "Option B",
            "C": "Both A and B",
            "D": "Option D",
        }
        result, _, _ = shuffle_options(options, "A", seed=42)
        assert result == options

    def test_label_refs_preserved_across_calls(self) -> None:
        """Verify label-refs unshuffled across seeds."""
        options = {
            "A": "X",
            "B": "Y",
            "C": "Either A or B",
            "D": "Z",
        }
        r1, _, _ = shuffle_options(options, "A", seed=42)
        r2, _, _ = shuffle_options(options, "A", seed=99)
        assert r1 == r2 == options

    def test_deterministic_same_seed_same_output(self) -> None:
        """Verify same seed produces identical shuffles."""
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1: dict[str, str]
        a1: str
        r1, a1, _ = shuffle_options(options, "B", seed=42)
        r2: dict[str, str]
        a2: str
        r2, a2, _ = shuffle_options(options, "B", seed=42)
        assert r1 == r2
        assert a1 == a2

    def test_different_seed_different_output(self) -> None:
        """Verify different seeds produce different shuffles."""
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1: dict[str, str]
        r1, _, _ = shuffle_options(options, "B", seed=42)
        r2: dict[str, str]
        r2, _, _ = shuffle_options(options, "B", seed=43)
        assert r1 != r2

    def test_seed_none_returns_unshuffled(self) -> None:
        """Verify seed=None disables shuffling."""
        options = {"A": "X", "B": "Y", "C": "Z"}
        result, answer, _ = shuffle_options(options, "B", seed=None)
        assert result == options

    def test_seed_none_preserves_answer(self) -> None:
        """Verify answer key unchanged when seed=None."""
        options = {"A": "X", "B": "Y", "C": "Z"}
        _, answer, _ = shuffle_options(options, "B", seed=None)
        assert answer == "B"

    def test_seed_minus_one_non_deterministic(self) -> None:
        """Verify seed=-1 produces non-deterministic shuffles."""
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        results = {
            tuple(r.items())
            for r, _, _ in [shuffle_options(options, "A", seed=-1) for _ in range(20)]
        }
        assert len(results) > 1

    def test_row_id_changes_shuffle(self) -> None:
        """Verify row_id changes shuffle with same seed."""
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1, _, _ = shuffle_options(options, "A", seed=42, row_id=1)
        r2, _, _ = shuffle_options(options, "A", seed=42, row_id=2)
        vals1 = list(r1.values())
        vals2 = list(r2.values())
        assert vals1 != vals2

    def test_row_id_consistent_with_same_value(self) -> None:
        """Verify same row_id and seed produce identical shuffles."""
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1, _, _ = shuffle_options(options, "A", seed=42, row_id=42)
        r2, _, _ = shuffle_options(options, "A", seed=42, row_id=42)
        assert r1 == r2

    def test_int_answer_choice_dict(self) -> None:
        """Verify int answer index maps correctly with dict options."""
        options = {"A": "X", "B": "Y", "C": "Z"}
        result, answer, _ = shuffle_options(options, 1, seed=0)
        assert result[answer] == "Y"

    def test_int_answer_choice_list(self) -> None:
        """Verify int answer index maps correctly with list options."""
        options = ["X", "Y", "Z"]
        labels = ["A", "B", "C"]
        result, answer, _ = shuffle_options(options, 1, labels=labels, seed=0)
        idx = labels.index(answer)
        assert result[idx] == "Y"

    def test_int_answer_out_of_range_raises(self) -> None:
        """Verify answer index out of range raises ValueError."""
        options = {"A": "X", "B": "Y"}
        with pytest.raises(ValueError, match="out of range"):
            shuffle_options(options, 5, seed=0)

    def test_digit_answer_choice_resolves_to_index(self) -> None:
        """Verify digit-string '2' resolves to correct letter."""
        options = {"A": "X", "B": "Y", "C": "Z"}
        result, answer, _ = shuffle_options(options, "2", seed=None)
        assert answer == "B"

    def test_invalid_str_answer_raises(self) -> None:
        """Verify unknown answer letter errors."""
        options = {"A": "X", "B": "Y"}
        with pytest.raises((ValueError, IndexError)):
            shuffle_options(options, "Z", seed=42)

    def test_invalid_answer_choice_raises(self) -> None:
        """Verify non-alpha, non-numeric answer raises ValueError."""
        options = {"A": "X", "B": "Y", "C": "Z"}
        with pytest.raises(ValueError, match="not found or invalid"):
            shuffle_options(options, "...", seed=None)

    def test_single_option_stays_unchanged(self) -> None:
        """Verify single option returned unchanged."""
        options = {"A": "Only one"}
        result, answer, _ = shuffle_options(options, "A", seed=42)
        assert result == {"A": "Only one"}
        assert answer == "A"

    def test_two_options_shuffled(self) -> None:
        """Verify two options shuffle and preserve keys."""
        options = {"A": "First", "B": "Second"}
        result, answer, _ = shuffle_options(options, "A", seed=42)
        assert set(result.keys()) == {"A", "B"}
        assert result[answer] == "First"

    def test_empty_options_raises(self) -> None:
        """Verify empty option dict raises IndexError or ValueError."""
        with pytest.raises((IndexError, ValueError)):
            shuffle_options({}, "A")

    def test_none_option_text_handled(self) -> None:
        """Verify None option text does not crash."""
        options = {"A": None, "B": "Value"}  # type: ignore[dict-item]
        result, answer, _ = shuffle_options(options, "B", seed=42)
        assert result is not None

    def test_answer_choice_not_in_options_after_na_filter(self) -> None:
        """Verify shuffle handles answer among remaining options after N/A filter."""
        options = {"A": "X", "B": "N/A"}
        result, answer, _ = shuffle_options(options, "A", seed=42)
        assert result is not None

    def test_non_collection_options_hashes_via_str(self) -> None:
        """Verify hash falls back to str() for non-list/dict inputs."""
        result = _stable_options_hash("just a string")
        assert isinstance(result, int)
        assert result > 0

    @pytest.mark.parametrize(
        "anchor_text",
        [
            "All of the above",
            "None of the above",
            "All of the following",
            "None of these",
            "Both of the above",
            "Some of these",
            "All of the above statements",
            "None of the above options",
        ],
    )
    def test_anchor_pattern_matches(self, anchor_text: str) -> None:
        """ANCHOR matches common anchor phrases."""
        assert ANCHOR.search(anchor_text)

    def test_anchor_does_not_match_ordinary_text(self) -> None:
        """Verify ANCHOR does not match ordinary text."""
        assert not ANCHOR.search("This is a regular option")

    def test_anchor_does_not_match_random_words(self) -> None:
        """Verify ANCHOR does not match generic anchor-word phrases."""
        assert not ANCHOR.search("All things considered")

    @pytest.mark.parametrize(
        "ref_text",
        [
            "Both A and B",
            "Either A or B",
            "Neither A nor B",
            "Both A and B and C",
            "Both A & B",
            "Only A and B",
            "Both A, B",
        ],
    )
    def test_label_ref_pattern_matches(self, ref_text: str) -> None:
        """LABEL_REF matches label-reference phrases."""
        assert LABEL_REF.search(ref_text)

    def test_label_ref_matches_with_and_or(self) -> None:
        """Verify LABEL_REF matches 'Both A and/or B'."""
        assert LABEL_REF.search("Both A and/or B")

    def test_label_ref_matches_parenthetical_labels(self) -> None:
        """Verify LABEL_REF matches parenthetical labels."""
        assert LABEL_REF.search("Both (A) and (B)")

    def test_label_ref_matches_bracket_labels(self) -> None:
        """Verify LABEL_REF matches bracket labels."""
        assert LABEL_REF.search("Both [A] and [B]")

    def test_label_ref_does_not_match_single_label(self) -> None:
        """Verify LABEL_REF requires multiple labels."""
        assert not LABEL_REF.search("Option A")

    def test_label_ref_does_not_match_plain_text(self) -> None:
        """Verify LABEL_REF does not match plain text."""
        assert not LABEL_REF.search("This is a random option")
