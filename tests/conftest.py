"""Collection hook: auto-apply markers by directory, enforce socket rules.

This top-level ``tests/conftest.py`` participates in collection for both the
``tests/unit/`` and ``tests/integration/`` trees via ``pytest_collection_modifyitems``.

- Items under ``tests/integration/`` receive ``integration`` + ``enable_socket``
  markers automatically, and ``flaky(reruns=3, ...)`` when ``pytest-rerunfailures``
  is installed.  Manual ``integration`` / ``enable_socket`` markers inside the
  integration tree are rejected (the directory is the single source of truth).
- Items outside ``tests/integration/`` that carry ``enable_socket``,
  ``socket_enabled``, or ``integration`` markers/fixtures are rejected
  (unit tests must mock clients, never request real sockets).
- Integration tests that require missing service configuration (checked against
  process env and repo-root ``.env`` via ``tests/_env.py``) are skipped with a
  descriptive reason.
"""

from __future__ import annotations

import pytest

from tests._env import is_configured, required_config_for


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-apply markers by directory; reject misplaced marker usage."""
    integration_root = config.rootpath / "tests" / "integration"
    has_rerunfailures = config.pluginmanager.hasplugin("rerunfailures")

    for item in items:
        is_integration = item.path.is_relative_to(integration_root)

        if not is_integration:
            _reject_unit_marker(item, "enable_socket")
            _reject_unit_marker(item, "integration")
            _reject_socket_fixture(item)
            continue

        # Integration item: markers are auto-applied by directory; ban manual ones.
        _reject_manual_marker_in_integration(item, "integration")
        _reject_manual_marker_in_integration(item, "enable_socket")

        item.add_marker(pytest.mark.integration)
        item.add_marker(pytest.mark.enable_socket)
        if has_rerunfailures:
            item.add_marker(
                pytest.mark.flaky(
                    reruns=3,
                    reruns_delay=30,
                    rerun_except=[AssertionError],
                )
            )

        try:
            required = required_config_for(item.path.as_posix(), item.path.name)
        except RuntimeError as exc:
            raise pytest.UsageError(f"{item.nodeid}: {exc}") from exc

        missing = [n for n in required if not is_configured(n)]
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"integration test skipped: missing required config "
                        f"{missing} (external service not configured for this env)"
                    ),
                )
            )


def _reject_unit_marker(item: pytest.Item, marker_name: str) -> None:
    """Raise ``UsageError`` if *marker_name* is present on a non-integration item."""
    if item.get_closest_marker(marker_name) is not None:
        raise pytest.UsageError(
            f"{item.nodeid}: {marker_name} is only allowed under "
            "tests/integration/. Mock external clients in unit tests; "
            "use allow_hosts only for intentional loopback-server tests."
        )


def _reject_socket_fixture(item: pytest.Item) -> None:
    """Raise ``UsageError`` if ``socket_enabled`` fixture is requested in unit test."""
    if "socket_enabled" in getattr(item, "fixturenames", ()):
        raise pytest.UsageError(
            f"{item.nodeid}: socket_enabled fixture is not allowed in unit tests."
        )


def _reject_manual_marker_in_integration(item: pytest.Item, marker_name: str) -> None:
    """Raise ``UsageError`` if *marker_name* is manually set on an integration item."""
    if item.get_closest_marker(marker_name) is not None:
        raise pytest.UsageError(
            f"{item.nodeid}: remove manual {marker_name} marker; "
            f"it is auto-applied to tests/integration/."
        )
