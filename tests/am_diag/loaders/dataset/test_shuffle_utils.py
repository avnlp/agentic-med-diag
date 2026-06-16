"""Unit tests for MCQ option shuffling logic."""

from __future__ import annotations

import pytest

from am_diag.loaders.dataset.shuffle_utils import (
    ANCHOR,
    LABEL_REF,
    shuffle_options,
)


class TestShuffleOptions:
    """Tests for shuffle_options()."""

    def test_dict_input_returns_dict_str_answer(self):
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answer, idx = shuffle_options(options, "B", seed=0)
        assert isinstance(result, dict)
        assert isinstance(answer, str)
        assert answer in options

    def test_dict_input_preserves_answer_text(self):
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answer, _ = shuffle_options(options, "B", seed=0)
        assert result[answer] == "Banana"

    def test_list_input_returns_list_str_answer(self):
        options = ["Apple", "Banana", "Cherry"]
        labels = ["A", "B", "C"]
        result, answer, idx = shuffle_options(options, "B", labels=labels, seed=0)
        assert isinstance(result, list)
        assert isinstance(answer, str)

    def test_list_input_without_labels_raises(self):
        options = ["A", "B", "C"]
        with pytest.raises(ValueError, match="labels must be provided"):
            shuffle_options(options, "B", seed=0)

    def test_list_input_wrong_label_count_raises(self):
        options = ["A", "B", "C"]
        with pytest.raises(ValueError, match="labels length"):
            shuffle_options(options, "B", labels=["A", "B"], seed=0)

    def test_list_input_preserves_answer_text(self):
        options = ["Apple", "Banana", "Cherry"]
        labels = ["A", "B", "C"]
        result, answer, _ = shuffle_options(options, "B", labels=labels, seed=0)
        idx = labels.index(answer)
        assert result[idx] == "Banana"

    def test_multi_answer_dict_returns_list(self):
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answers, idx = shuffle_options(options, ["A", "C"], seed=0)
        assert isinstance(result, dict)
        assert isinstance(answers, list)
        assert len(answers) == 2

    def test_multi_answer_all_labels_valid(self):
        options = {"A": "Apple", "B": "Banana", "C": "Cherry"}
        result, answers, _ = shuffle_options(options, ["A", "C"], seed=0)
        assert all(a in result for a in answers)

    def test_multi_answer_list_options(self):
        options = ["Apple", "Banana", "Cherry"]
        labels = ["A", "B", "C"]
        result, answers, _ = shuffle_options(options, ["A", "C"], labels=labels, seed=0)
        assert isinstance(result, list)
        assert isinstance(answers, list)
        assert len(answers) == 2

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
    def test_anchor_pattern_matches(self, anchor_text):
        assert ANCHOR.search(anchor_text)

    def test_anchors_preserved_in_place(self):
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

    def text_anchors_in_list_input(self):
        options = ["Option X", "All of the above", "Option Z"]
        labels = ["A", "B", "C"]
        result, _, _ = shuffle_options(options, "A", labels=labels, seed=42)
        assert result[1] == "All of the above"

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
    def test_label_ref_pattern_matches(self, ref_text):
        assert LABEL_REF.search(ref_text)

    def test_label_refs_skip_shuffle(self):
        options = {
            "A": "Option A",
            "B": "Option B",
            "C": "Both A and B",
            "D": "Option D",
        }
        result, _, _ = shuffle_options(options, "A", seed=42)
        assert result == options

    def test_label_refs_preserved_across_calls(self):
        options = {
            "A": "X",
            "B": "Y",
            "C": "Either A or B",
            "D": "Z",
        }
        r1, _, _ = shuffle_options(options, "A", seed=42)
        r2, _, _ = shuffle_options(options, "A", seed=99)
        assert r1 == r2 == options

    def test_deterministic_same_seed_same_output(self):
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1: dict[str, str]
        a1: str
        r1, a1, _ = shuffle_options(options, "B", seed=42)
        r2: dict[str, str]
        a2: str
        r2, a2, _ = shuffle_options(options, "B", seed=42)
        assert r1 == r2
        assert a1 == a2

    def test_different_seed_different_output(self):
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1: dict[str, str]
        r1, _, _ = shuffle_options(options, "B", seed=42)
        r2: dict[str, str]
        r2, _, _ = shuffle_options(options, "B", seed=43)
        assert r1 != r2

    def test_seed_none_returns_unshuffled(self):
        options = {"A": "X", "B": "Y", "C": "Z"}
        result, answer, _ = shuffle_options(options, "B", seed=None)
        assert result == options

    def test_seed_none_preserves_answer(self):
        options = {"A": "X", "B": "Y", "C": "Z"}
        _, answer, _ = shuffle_options(options, "B", seed=None)
        assert answer == "B"

    def test_seed_minus_one_non_deterministic(self):
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        results = {
            tuple(r.items())
            for r, _, _ in [shuffle_options(options, "A", seed=-1) for _ in range(20)]
        }
        assert len(results) > 1

    def test_row_id_changes_shuffle(self):
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1, _, _ = shuffle_options(options, "A", seed=42, row_id=1)
        r2, _, _ = shuffle_options(options, "A", seed=42, row_id=2)
        vals1 = list(r1.values())
        vals2 = list(r2.values())
        assert vals1 != vals2

    def test_row_id_consistent_with_same_value(self):
        options = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
        r1, _, _ = shuffle_options(options, "A", seed=42, row_id=42)
        r2, _, _ = shuffle_options(options, "A", seed=42, row_id=42)
        assert r1 == r2

    def test_int_answer_choice_dict(self):
        options = {"A": "X", "B": "Y", "C": "Z"}
        result, answer, _ = shuffle_options(options, 1, seed=0)
        assert result[answer] == "Y"

    def test_int_answer_choice_list(self):
        options = ["X", "Y", "Z"]
        labels = ["A", "B", "C"]
        result, answer, _ = shuffle_options(options, 1, labels=labels, seed=0)
        idx = labels.index(answer)
        assert result[idx] == "Y"

    def test_int_answer_out_of_range_raises(self):
        options = {"A": "X", "B": "Y"}
        with pytest.raises(ValueError, match="out of range"):
            shuffle_options(options, 5, seed=0)

    def test_single_option_stays_unchanged(self):
        options = {"A": "Only one"}
        result, answer, _ = shuffle_options(options, "A", seed=42)
        assert result == {"A": "Only one"}
        assert answer == "A"

    def test_two_options_shuffled(self):
        options = {"A": "First", "B": "Second"}
        result, answer, _ = shuffle_options(options, "A", seed=42)
        assert set(result.keys()) == {"A", "B"}
        assert result[answer] == "First"

    def test_invalid_str_answer_raises(self):
        options = {"A": "X", "B": "Y"}
        with pytest.raises((ValueError, IndexError)):
            shuffle_options(options, "Z", seed=42)

    def test_empty_options_raises(self):
        with pytest.raises((IndexError, ValueError)):
            shuffle_options({}, "A")

    def test_none_option_text_handled(self):
        options = {"A": None, "B": "Value"}  # type: ignore[dict-item]
        result, answer, _ = shuffle_options(options, "B", seed=42)
        assert result is not None

    def test_answer_choice_not_in_options_after_na_filter(self):
        options = {"A": "X", "B": "N/A"}
        result, answer, _ = shuffle_options(options, "A", seed=42)
        assert result is not None


class TestAnchorRegex:
    """Targeted tests for the ANCHOR regex pattern."""

    def test_matches_all_of_the_above(self):
        assert ANCHOR.search("All of the above")

    def test_matches_none_of_the_following(self):
        assert ANCHOR.search("None of the following")

    def test_matches_both_of_these(self):
        assert ANCHOR.search("Both of these")

    def test_matches_some_of_the_options(self):
        assert ANCHOR.search("Some of the options")

    def test_matches_all_of_the_choices(self):
        assert ANCHOR.search("All of the choices")

    def test_matches_neither_of_them(self):
        assert ANCHOR.search("Neither of them")

    def test_matches_all_of_the_statements(self):
        assert ANCHOR.search("All of the statements")

    def test_does_not_match_ordinary_text(self):
        assert not ANCHOR.search("This is a regular option")

    def test_does_not_match_random_words(self):
        assert not ANCHOR.search("All things considered")


class TestLabelRefRegex:
    """Targeted tests for the LABEL_REF regex pattern."""

    def test_matches_both_a_and_b(self):
        assert LABEL_REF.search("Both A and B")

    def test_matches_either_a_or_b(self):
        assert LABEL_REF.search("Either A or B")

    def test_matches_neither_a_nor_b(self):
        assert LABEL_REF.search("Neither A nor B")

    def test_matches_three_labels(self):
        assert LABEL_REF.search("Both A and B and C")

    def test_matches_with_and_or(self):
        assert LABEL_REF.search("Both A and/or B")

    def test_matches_only_a_and_b(self):
        assert LABEL_REF.search("Only A and B")

    def test_matches_parenthetical_labels(self):
        assert LABEL_REF.search("Both (A) and (B)")

    def test_matches_bracket_labels(self):
        assert LABEL_REF.search("Both [A] and [B]")

    def test_does_not_match_single_label(self):
        assert not LABEL_REF.search("Option A")

    def test_does_not_match_plain_text(self):
        assert not LABEL_REF.search("This is a random option")
