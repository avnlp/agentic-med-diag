---
name: writing-tests
description: Testing rules for unit and integration tests. Use when adding, changing, or fixing pytest tests under tests/.
---

# Testing Rules

Use these rules for tests in this repository.

## Core rules

- Never write unit tests after you write code.
- Highly prefer E2E or integration tests as the sole testing mechanism. Use them to verify complex features work. At the end of those tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code.
- A test that breaks under a behavior-preserving refactor is asserting implementation, not behavior. Do not add it.
- Never delete or weaken a failing test to make the suite pass. Fix the code, or ask.
- Do not add tests that restate the code, assert mock call shapes, grep source or docs text, compute the expected value with the code under test, or only keep a test-only export or seam alive.

## Organization

- Use `pytest` and `pytest-asyncio` for tests, including async tests.
- Tests mirror the `am_diag/` package structure under `tests/am_diag/`.
  Integration tests sit beside the unit tests in `test_<name>_integration.py`.
- Use one `Test<Feature>` class per `test_<feature>.py` file.
- `pytest-asyncio` runs in auto mode. Do not add `@pytest.mark.asyncio` to async
  tests.
- Pytest runs with `-m 'not integration'` by default, so integration tests run
  only through `make test-integration`.
- Tests run in parallel. Avoid shared mutable state between
  test classes.
- Use descriptive, behavior-oriented test names such as
  `test_returns_empty_when_no_results`.
- Test the public interface and documented behavior, not private implementation
  details. Avoid accessing private members in tests unless there is no public
  seam and adding one would be worse for the design.
- Follow nearby test patterns before introducing new fixtures or helpers.
- For bug fixes, extend the E2E or integration test that covers the behavior so
  it fails without the fix. Add a unit test only when the core rules allow it.
- One focused test that fails without the fix is better than many shallow tests
  that do not prove behavior.

## Isolation

- Unit tests are the exception. When you write one, it must mock external
  dependencies such as Neo4j, Qdrant, Weaviate, Hugging Face, and LLM APIs.
- Unit tests must not enable sockets. If a unit test hits `SocketBlockedError`,
  fix the mock rather than enabling network access.
- `pytest-socket` disables network by default with
  `--disable-socket --allow-unix-socket`.
- Mark every integration test with both `@pytest.mark.integration` and
  `@pytest.mark.enable_socket`. Do not add these markers to unit tests.
- Patch external drivers at the driver boundary used by production code.
- For Neo4j unit tests, patch `am_diag.db.graph.neo4j.AsyncGraphDatabase.driver`.
- Avoid shared mutable state between test classes because tests run in parallel.

## Maintenance

- Tests must pass after every behavior change.
- Tests for changed behavior should fail when the behavior is broken. Do not add
  tests that merely exercise code without meaningful assertions.
- Refactoring internals should not require test changes. If it does, the tests
  may be too coupled to implementation details.
- When changing public APIs, update tests for the new public contract.
- When deleting behavior, delete or update the corresponding tests.
- Do not delete, skip, or weaken a failing test to hide the failure. Use
  `@pytest.mark.skip(reason="...")` or `@pytest.mark.xfail` only for a known,
  documented reason the user has accepted.
- Do not write assertion-free tests only to increase coverage.
- Target 80-90% coverage for core/domain logic; do not chase 100% coverage.

## Commands

- Use `uv run` for Python commands.
- Use focused pytest commands while iterating, then run the narrowest broader
  check that proves the change.
- Example focused command:
  `uv run pytest tests/am_diag/db/graph/test_client.py -k client`.
