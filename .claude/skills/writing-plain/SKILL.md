---
name: writing-plain
description: Writes and edits plain prose. Use for agent replies, documentation, instruction files, comments, docstrings, reports, commit messages, and PR descriptions, or when asked to simplify, clarify, or deslopify text.
license: MIT
---

# Plain writing

Use plain, literal language that readers can understand in one pass. Apply this skill alongside the relevant task skill whenever you draft or revise prose.

## Scope and precedence

The numbered rules below adapt docwriter's plain-writing skill. They take precedence over the STE guidance in this file. Task skills govern technical content, document structure, and required formatting, not competing prose styles. Preserve technical meaning, uncertainty, requirement strength, and necessary qualifications.

Apply prose rules to comments and docstrings when the task edits them, including examples inside code fences. Preserve code syntax, identifiers, commands, paths, URLs, exact quotations, official names, and required notices. Do not rewrite BAML prompts, test fixtures, expected outputs, protocol payloads, or generated files solely to satisfy prose style. English text used as data is not automatically editable prose.

## Word choice and tone

1. Use simple, everyday words when they express the meaning precisely. Avoid fancy synonyms and overused AI wording such as "delve", "tapestry", "robust", and "leverage".
2. Use familiar language instead of jargon or invented shorthand. Keep established technical terms when they are most precise, and briefly define them when the audience may not know them.
3. Remove puffery and empty emphasis. Replace claims such as "pivotal" or "carries weight" with a concrete fact, or cut the claim.
4. Use consistent terminology. Use one name for one concept, but preserve distinctions among different concepts. A document, file, and database record can represent different objects.
5. Allow contractions when they sound natural. Do not require them.
6. Avoid invented hyphenated adjectives. Established compounds such as "well-crafted" are fine.
7. Keep writing descriptive and explanatory. Use literal headings, labels, and summaries instead of slogans or clever phrases.

## Sentences and paragraphs

8. Write complete sentences with subjects and verbs. Join related ideas with ordinary conjunctions, and separate unrelated ideas. Do not join unrelated clauses with colons or semicolons.
9. Present workflows in order. Give each step its own sentence with "First", "Second", and "Third", or use numbered steps. Semicolons may separate steps in a closely related workflow.
10. Start a paragraph with its main point, then give supporting facts or examples. Introduce further support with a plain connective when needed.
11. Use at most two related clauses in an ordinary prose sentence. Do not give three or more example sentences in a row to support the same point. Use an ordered sequence or brief list when more points are necessary.
12. Prefer full explanatory sentences over a stack of short, punchy sentences. Keep related ideas together without padding. Short lines are fine in labeled briefs, bullets, and ordered sequences.
13. State exactly what changes, who acts, or how something works. Remove side clauses that add no information, and retain qualifications that affect meaning.
14. In chat, give brief problem context before explaining a change. Write for a reader who may have forgotten the context. Do not repeat the full history. Essays do not need chat-style setup.

## Punctuation and formatting

15. Do not use em dashes, en dashes, or middle dots in authored prose. Write ranges with "to", and separate ideas with periods, commas, or conjunctions. Ordinary hyphens and required syntax remain valid.
16. Use colons to introduce lists or short labels in updates, briefs, and PR descriptions. Do not use colons to join clauses in ordinary prose.
17. Use straight quotes rather than curly quotes.
18. Use sentence case in headings and plain formatting. Do not use bold for decoration.
19. Keep explanatory lists to three or four points where practical, and group additional points. Use lists and tables sparingly in essays. Preserve complete procedures, API inventories, and required checklists even when they need more entries.

## Patterns to avoid

20. Name the real actor instead of giving objects human intent. Factual system verbs are valid, such as "The API returns JSON" or "The job writes the file".
21. Explain the actual subject without analogies, imagery, or metaphors. Preserve official names and established technical terms.
22. State what something is instead of using "not just X, but Y" or "not only X, but Y". Plain refusals and factual corrections are fine.
23. State the problem directly instead of stacking rhetorical questions.
24. Replace demonstrative pronouns such as "this", "that", "these", and "those" with explicit subjects. Do not open a paragraph with a sentence containing a demonstrative. The grammatical connector "that", as in "Check that the file exists", is not a demonstrative.
25. Start with the point instead of announcing how many points will follow. Use a list when the content needs multiple entries.

## Compatible STE guidance

Use the following guidance for technical explanations and procedures when it agrees with the numbered rules:

- Use clear actors and imperative procedural steps. Allow passive voice for states, results, or unknown actors when it is clearer. Do not invent an actor.
- Put a condition before a command when the condition determines whether the action is valid. Separate actions from explanatory notes when readers need to follow a procedure. Prefer one action per step, with related simultaneous actions allowed.
- State specific quantities and units when known. Keep necessary articles, subjects, verbs, and connecting words. Rewrite long noun groups with words such as "of" or "for" when that improves clarity.
- Use American spelling outside protected text. For actual safety instructions, use "WARNING" for risk of injury or death and "CAUTION" for risk of damage to objects. State the action and the risk. Do not force safety labels onto ordinary software callouts.

Do not enforce the STE dictionary, approved parts of speech, hard word counts, a six-sentence paragraph limit, or bans on contractions, semicolons, phrasal verbs, verb tenses, and "-ing" forms. Keep modal verbs such as "may", "might", and "should" when they express the correct possibility or obligation. Never replace "may" with "must" just to simplify wording.

The STE source uses ASD-STE100 Issue 7. The repository adaptation does not certify compliance with ASD-STE100, and it does not include the ASD word list or checker.

## Repository format exceptions

Keep the established formats below without expanding every entry into a full explanatory sentence:

- Google docstring sections such as `Args:`, `Returns:`, and `Raises:`, concise field descriptions, and one-line descriptive or imperative summaries.
- Headings, table labels, conventional commit prefixes, and short imperative commit subjects without a final period.
- Markdown syntax, YAML front matter, Python quote delimiters, and required safety or attribution labels.

Follow the repository's 88-character limit for Python comments and docstrings. Wrap longer docstring descriptions naturally without dropping necessary facts. Do not hard-wrap Markdown prose to that limit.

## Deslopify

When the user invokes `/writing-plain deslopify`, says `/plain-writing deslopify`, or asks to deslopify text, rewrite the supplied text. If no text follows the request, rewrite the previous agent response. Return only the rewrite.

Start with the main conclusion, then give the background and explain how it works in logical order. Write for a technical reader without project context. Use established terminology and define unfamiliar terms. Preserve facts, caveats, numbers, technical contracts, and protected text.

## Review before delivery

Check the draft against every applicable numbered rule. Confirm that workflows remain ordered and complete, terminology stays consistent, and the rewrite preserves facts and qualifications. Check format exceptions rather than treating them as prose errors. Review protected content for accidental changes.

## Sources and license

Adapted from [docwriter's plain-writing skill](https://github.com/docwriter-org/plain-writing-skill/blob/f0d3630983ac7a82aa580f1c1509d72df739ee12/skills/plain-writing/SKILL.md), copyright (c) 2026 Shreya Shankar, and [0xpili's simplified-technical-english skill](https://github.com/0xpili/simplified-technical-english/blob/1e148d670cba46685ad2b4c3f2354a637a7fdbbe/SKILL.md), copyright (c) 2026 0xpili.

The adaptation condenses docwriter's examples, preserves its rule numbering, adds subordinate STE guidance, and defines repository scope and format exceptions. Both skill texts use the MIT license. Their full notices are in [LICENSE.md](LICENSE.md), which must accompany copies of this skill. The ASD dictionary is separate from those MIT grants and is not included.
