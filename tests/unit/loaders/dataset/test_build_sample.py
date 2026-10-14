"""Central tests for `MCQDataset._build_sample`.

The shuffle-and-build invariant lives once in `MCQDataset._build_sample`, so it
is tested once here rather than duplicated across every loader. Each loader's
own tests cover only its `_extract` (and `_rows`) parse logic.
"""

from __future__ import annotations

from typing import Any, ClassVar

from am_diag.loaders.dataset.base import ExtractedMCQ, MCQDataset


class _StubMCQDataset(MCQDataset):
    """Minimal concrete loader: `_extract` is unused; we call `_build_sample`."""

    dataset_name: ClassVar[str] = "stub"
    hf_repo: ClassVar[str] = "stub/repo"

    def _extract(self, idx: int, record: Any) -> ExtractedMCQ | None:  # noqa: ARG002
        return None


def _build(
    extracted: ExtractedMCQ, *, shuffle: bool, idx: int = 0, split: str = "test"
):
    loader = _StubMCQDataset(shuffle_options=shuffle, shuffle_seed=42)
    return loader._build_sample(idx, extracted, split)


def _single(**overrides: Any) -> ExtractedMCQ:
    kwargs: dict[str, Any] = {
        "question_stem": "What is X?",
        "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "answer_keys": ["B"],
    }
    kwargs.update(overrides)
    return ExtractedMCQ(**kwargs)


class TestBuildSampleNoShuffle:
    """Tests for _build_sample with shuffle_options=False."""

    def test_options_unchanged_and_no_original(self):
        """Test that options are unchanged and options_original is None."""
        e = _single()
        s = _build(e, shuffle=False)
        assert s.options == e.options
        assert s.options_original is None

    def test_invariants_hold(self):
        """Test that answer, answer_key, and answer_text invariants hold."""
        s = _build(_single(), shuffle=False)
        assert s.answer == s.answer_key == s.answer_keys[0]
        assert s.answer_text == s.options[s.answer_key] == s.answer_texts[0]

    def test_sample_id_falls_back_to_idx(self):
        """Test that sample_id defaults to the string version of the row index."""
        s = _build(_single(), shuffle=False, idx=7)
        assert s.sample_id == "7"

    def test_sample_id_uses_provided(self):
        """Test that a provided sample_id is used instead of the row index."""
        s = _build(_single(sample_id="q99"), shuffle=False)
        assert s.sample_id == "q99"

    def test_split_is_recorded(self):
        """Test that the split argument is stored on the sample."""
        s = _build(_single(), shuffle=False, split="validation")
        assert s.split == "validation"

    def test_metadata_passed_through(self):
        """Test that metadata dict is passed through to the sample unchanged."""
        s = _build(_single(metadata={"k": "v"}), shuffle=False)
        assert s.metadata == {"k": "v"}


class TestBuildSampleShuffle:
    """Tests for _build_sample with shuffle_options=True."""

    def test_options_original_preserved(self):
        """Test that options_original captures the pre-shuffle ordering."""
        e = _single()
        s = _build(e, shuffle=True)
        assert s.options_original == e.options

    def test_invariants_hold_after_shuffle(self):
        """Test that answer invariants are maintained after shuffling."""
        s = _build(_single(), shuffle=True)
        assert s.answer == s.answer_key == s.answer_keys[0]
        assert s.answer_text == s.options[s.answer_key] == s.answer_texts[0]

    def test_correct_answer_text_preserved_through_shuffle(self):
        """Test that correct answer text is preserved regardless of shuffle ordering."""
        e = _single()
        original_text = e.options[e.answer_keys[0]]
        s = _build(e, shuffle=True)
        assert s.answer_text == original_text

    def test_deterministic_with_seed(self):
        """Test that shuffle is deterministic when the same seed is used."""
        s1 = _build(_single(), shuffle=True)
        s2 = _build(_single(), shuffle=True)
        assert s1.options == s2.options
        assert s1.answer_key == s2.answer_key

    def test_seeded_shuffle_changes_option_order(self):
        """Test that a seeded shuffle reorders options for some rows."""
        e = _single()
        orders = {
            tuple(_build(e, shuffle=True, idx=i).options.values()) for i in range(20)
        }
        assert len(orders) > 1

    def test_anchor_option_stays_in_place(self):
        """Test that anchor options like 'None of the above' stay at D."""
        e = _single(
            options={"A": "alpha", "B": "beta", "C": "gamma", "D": "None of the above"},
            answer_keys=["A"],
        )
        s = _build(e, shuffle=True)
        assert s.options["D"] == "None of the above"


class TestBuildSampleMultiAnswer:
    """Tests for _build_sample with multi-answer (is_multi_answer=True)."""

    def test_multi_keys_and_texts(self):
        """Test that multiple answer keys and texts are stored on the sample."""
        e = ExtractedMCQ(
            question_stem="Pick all",
            options={"A": "Red", "B": "Green", "C": "Blue"},
            answer_keys=["A", "C"],
            is_multi_answer=True,
        )
        s = _build(e, shuffle=False)
        assert s.answer_keys == ["A", "C"]
        assert s.answer_texts == ["Red", "Blue"]
        assert s.is_multi_answer is True

    def test_multi_invariants_hold_after_shuffle(self):
        """Test that multi-answer invariants (key-text mapping) hold after shuffling."""
        e = ExtractedMCQ(
            question_stem="Pick all",
            options={"A": "Red", "B": "Green", "C": "Blue", "D": "Yellow"},
            answer_keys=["A", "C"],
            is_multi_answer=True,
        )
        s = _build(e, shuffle=True)
        assert s.is_multi_answer is True
        assert len(s.answer_keys) == 2
        for i, k in enumerate(s.answer_keys):
            assert s.answer_texts[i] == s.options[k]
        # The two correct texts are preserved through the shuffle.
        assert set(s.answer_texts) == {"Red", "Blue"}

    def test_multi_answer_prompt_instruction(self):
        """Test that multi-answer prompts use plural 'letter(s)' phrasing."""
        e = ExtractedMCQ(
            question_stem="Pick all",
            options={"A": "Red", "B": "Green"},
            answer_keys=["A"],
            is_multi_answer=True,
        )
        s = _build(e, shuffle=False)
        assert "letter(s)" in s.question


class TestBuildSamplePrompt:
    """Tests for the generated prompt structure in _build_sample."""

    def test_prompt_structure(self):
        """Test that prompt contains Question, Choices, and Answer sections."""
        s = _build(_single(), shuffle=False)
        assert s.question.startswith("Question:")
        assert "Choices:" in s.question
        assert "Answer with" in s.question

    def test_single_answer_instruction(self):
        """Test that single-answer prompts use 'only the letter' phrasing."""
        s = _build(_single(), shuffle=False)
        assert "only the letter" in s.question
