"""Deterministic MCQ option shuffling with anchor preservation."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, overload


ANCHOR = re.compile(
    r"""
    \b
    (?:all|none|some|both|neither)
    \s+
    (?:of\s+(?:the\s+)?)?
    (?:
        above
        |following
        |these
        |choices?
        |options?
        |answers?
        |statements?
        |responses?
        |listed
        |apply
        |applicable
        |them
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LABEL_TOKEN = (
    r"(?:\((?-i:[A-Z0-9]+)\)|\[(?-i:[A-Z0-9]+)\]"
    r"|\b(?-i:[A-Z])(?:[)\.:])?"
    r"|\b\d+(?:[)\.:])?)"
)
LABEL_REF = re.compile(
    rf"""
    (?:\b(?:both|either|neither|only)\b\s+)?
    {_LABEL_TOKEN}
    (?:\s*[,&/]+\s*|\s+(?:and/or|and|or|nor)\s+)
    {_LABEL_TOKEN}
    (?:
        (?:\s*[,&/]+\s*|\s+(?:and/or|and|or|nor)\s+)
        {_LABEL_TOKEN}
    )*
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _stable_options_hash(options: list[str] | dict[str, str] | Any) -> int:
    r"""Deterministic hash of the option content for per-row seed mixing.

    Uses SHA-256 truncated to 64 bits. The `\x1e` / `\x1f` separators
    are ASCII record/unit separators that cannot appear in normal text,
    ensuring list vs. dict encoding is unambiguous.

    Args:
        options: Option texts as a list or a dict mapping labels to texts.

    Returns:
        A 64-bit unsigned integer derived from the option content.
    """
    if isinstance(options, list):
        serialized = "\x1e".join(["" if o is None else str(o) for o in options])
    elif isinstance(options, dict):
        serialized = "\x1f".join([f"{k}\x1e{options[k]}" for k in sorted(options)])
    else:
        serialized = str(options)
    return int(hashlib.sha256(serialized.encode()).hexdigest(), 16) & ((1 << 64) - 1)


@overload
def shuffle_options(
    options: dict[str, str],
    answer_choice: str | int,
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
) -> tuple[dict[str, str], str, int]: ...


@overload
def shuffle_options(
    options: dict[str, str],
    answer_choice: list[str],
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
) -> tuple[dict[str, str], list[str], int]: ...


@overload
def shuffle_options(
    options: list[str],
    answer_choice: str | int,
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
) -> tuple[list[str], str, int]: ...


@overload
def shuffle_options(
    options: list[str],
    answer_choice: list[str],
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
) -> tuple[list[str], list[str], int]: ...


def shuffle_options(  # noqa: PLR0912, PLR0915
    options: list[str] | dict[str, str],
    answer_choice: str | int | list[str],
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
) -> tuple[list[str] | dict[str, str], str | list[str], int]:
    """Randomize MCQ options while preserving anchor options in place.

    Anchor phrases like "All of the above" / "None of the following" stay in
    their original positions; only non-anchor segments are shuffled.

    Label reference detection: if any option references other option labels
    (e.g. "Both A and B"), shuffling is skipped entirely for that question
    to avoid breaking the references.

    Args:
        options: List of option texts or dict mapping labels to option texts.
        answer_choice: Original answer as 0-based index, label string,
            or list of label strings (for multi-answer questions).
        labels: Required label strings for list inputs (e.g. `["A","B","C"]`).
        seed: `None` → no shuffle, `-1` → non-deterministic,
            `>= 0` → deterministic.
        row_id: Mixed into seed for per-row variation.

    Returns:
        Tuple of `(shuffled_options, new_answer_label(s), new_answer_index)`.
        For multi-answer, `new_answer_label` is a list of labels
        (e.g. `["A", "C"]`) and `new_answer_index` is the first index.
    """
    if isinstance(options, dict):
        labels = list(options.keys())
        texts = [options[k] for k in labels]
        dict_mode = True
    else:
        texts = list(options)
        if labels is None:
            raise ValueError("labels must be provided when options is a list")
        if len(labels) != len(texts):
            raise ValueError(
                f"labels length ({len(labels)}) must match number of options "
                f"({len(texts)}) for list inputs",
            )
        dict_mode = False

    def norm_label(s: Any) -> str:
        """Normalise a label to its alphabetic/digit prefix in uppercase."""
        m = re.search(r"([A-Za-z]+|\d+)", str(s))
        return m.group(1).upper() if m else str(s).upper()

    def _resolve(ac: str | int) -> int:
        """Map an answer choice to a 0-based index into the options list."""
        if isinstance(ac, int):
            if not (0 <= ac < len(texts)):
                raise ValueError(
                    f"answer_choice={ac!r} is out of range for {len(texts)} options",
                )
            return ac
        wanted = norm_label(ac)
        for i, lab in enumerate(labels):
            if norm_label(lab) == wanted:
                return i
        if wanted.isalpha():
            return ord(wanted) - ord("A")
        if wanted.isdigit():
            return int(wanted) - 1
        raise ValueError(
            f"answer_choice={ac!r} not found or invalid among labels={labels}",
        )

    multi_mode = isinstance(answer_choice, list)
    if multi_mode:
        answer_idxs = [_resolve(ac) for ac in answer_choice]
    else:
        answer_idxs = [_resolve(answer_choice)]

    def _make_result(
        opts: list[str] | dict[str, str],
    ) -> tuple[list[str] | dict[str, str], str | list[str], int]:
        assert labels is not None
        if multi_mode:
            new_labels = [labels[i] for i in answer_idxs]
            return opts, new_labels, answer_idxs[0]
        return opts, labels[answer_idxs[0]], answer_idxs[0]

    if seed is None:
        return _make_result(
            dict(zip(labels, texts, strict=False)) if dict_mode else list(texts),
        )
    if seed == -1:
        rng = random.Random()
    else:
        if row_id is None:
            row_id = _stable_options_hash(options)
        mix = f"{seed}::{row_id}" if row_id is not None else f"{seed}"
        rng = random.Random(
            int(hashlib.sha256(mix.encode()).hexdigest(), 16) & ((1 << 64) - 1),
        )

    has_label_refs = any(LABEL_REF.search(t or "") for t in texts)
    if has_label_refs:
        return _make_result(
            dict(zip(labels, texts, strict=False)) if dict_mode else list(texts),
        )

    n = len(texts)
    anchors = [i for i, t in enumerate(texts) if ANCHOR.search(t or "")]
    blocks = []
    last = 0
    for a in anchors:
        if last < a:
            blocks.append((last, a))
        last = a + 1
    if last < n:
        blocks.append((last, n))

    index_map = list(range(n))
    for start, end in blocks:
        idxs = list(range(start, end))
        rng.shuffle(idxs)
        orig_txts = texts[start:end]
        orig_map = index_map[start:end]
        for off, dst in enumerate(idxs):
            texts[start + off] = orig_txts[dst - start]
            index_map[start + off] = orig_map[dst - start]

    answer_idxs = [index_map.index(i) for i in answer_idxs]

    return _make_result(dict(zip(labels, texts, strict=False)) if dict_mode else texts)
