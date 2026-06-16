"""MCQ prompt formatting utilities."""

from __future__ import annotations


def format_mcq_prompt(
    stem: str,
    options: dict[str, str],
    is_multi_answer: bool = False,
) -> str:
    """Build a combined MCQ prompt from question stem and options.

    Format::

        Question: <stem>
        Choices:
        A. <option text>
        B. <option text>
        ...
        Answer with only the letter of the correct option.

    For multi-answer questions (`is_multi_answer=True`) the instruction
    becomes:

        Answer with the letter(s) of the correct option(s) (e.g., 'A' or 'B,C').

    Args:
        stem: The raw question text (without answer choices).
        options: Dict mapping option letters to their text.
        is_multi_answer: Whether the question expects multiple correct answers.

    Returns:
        A formatted prompt string ready for the model.
    """
    choices = "\n".join(f"{k}. {v}" for k, v in options.items())
    instruction = (
        "Answer with the letter(s) of the correct option(s) (e.g., 'A' or 'B,C')."
        if is_multi_answer
        else "Answer with only the letter of the correct option."
    )
    return f"Question: {stem}\nChoices:\n{choices}\n{instruction}"
