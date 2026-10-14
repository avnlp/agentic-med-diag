"""Shared shuffle test mixin for MCQ dataset loader tests."""

from __future__ import annotations

from typing import Any


class ShuffleTestMixin:
    """Mixin providing 4 shared shuffle tests for MCQ dataset loaders.

    Subclasses must set:
        loader_class : The dataset loader class under test.

    Subclasses must define:
        make_row(**overrides) -> dict[str, Any] : Factory for a valid row.

    Subclasses may override:
        _loader_kwargs : Extra constructor kwargs (e.g. subjects for MMLUMed).
        _patch_and_load(patch_load_dataset, **loader_kwargs) : Custom patch logic.
    """

    loader_class: type | None = None

    def make_row(self, **overrides: Any) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def _loader_kwargs(self) -> dict[str, Any]:
        return {}

    @property
    def _shuffle_kwargs(self) -> dict[str, Any]:
        return {"shuffle_options": True, "shuffle_seed": 42}

    def _patch_and_load(
        self,
        patch_load_dataset: Any,
        **loader_kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError

    def test_shuffle_modifies_options_order(self, patch_load_dataset: Any) -> None:
        kwargs = {**self._loader_kwargs, **self._shuffle_kwargs}
        samples = self._patch_and_load(patch_load_dataset, **kwargs)
        assert len(samples) == 1
        assert samples[0].options_original is not None
        assert samples[0].options != samples[0].options_original

    def test_shuffle_preserves_original_in_options_original(
        self,
        patch_load_dataset: Any,
    ) -> None:
        kwargs = {**self._loader_kwargs, **self._shuffle_kwargs}
        samples = self._patch_and_load(patch_load_dataset, **kwargs)
        assert samples[0].answer_text in samples[0].options_original.values()

    def test_shuffle_invariant_holds(self, patch_load_dataset: Any) -> None:
        kwargs = {**self._loader_kwargs, **self._shuffle_kwargs}
        samples = self._patch_and_load(patch_load_dataset, **kwargs)
        assert samples[0].answer == samples[0].answer_key
        assert samples[0].answer_text == samples[0].options[samples[0].answer_key]

    def test_shuffle_deterministic_with_seed(self, patch_load_dataset: Any) -> None:
        kwargs = {**self._loader_kwargs, **self._shuffle_kwargs}
        s1 = self._patch_and_load(patch_load_dataset, **kwargs)
        s2 = self._patch_and_load(patch_load_dataset, **kwargs)
        assert s1[0].options == s2[0].options
        assert s1[0].answer_key == s2[0].answer_key
