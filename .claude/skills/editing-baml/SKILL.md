---
name: editing-baml
description: BAML source and generated client rules. Use when editing .baml files under am_diag/llm/baml_src/, regenerating the client, or changing BAML prompts, clients, or tests.
---

# Editing BAML

- Edit BAML sources under `am_diag/llm/baml_src/`. Never manually edit the
  generated client under `am_diag/llm/baml_client/`.
- After any `.baml` change, regenerate the client with `make baml-gen` (not a
  bare `baml-cli generate`), and commit the regenerated client.
- The generator targets `python/pydantic`; generated BAML types are Pydantic
  models.
- For model choice use current generally available models and verify against
  provider docs. Model IDs in [REFERENCE.md](REFERENCE.md) illustrate syntax
  only; do not copy them.

Read [REFERENCE.md](REFERENCE.md) for BAML syntax, types, prompts, clients, and
tests. Prefer the rules above and `https://docs.boundaryml.com` when they
disagree with it.
