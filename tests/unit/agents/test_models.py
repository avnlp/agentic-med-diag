"""build_chat_model — default, overrides, and None settings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from am_diag.agents.models import build_chat_model
from am_diag.agents.settings import AgentSettings


class TestBuildChatModel:
    """build_chat_model — default, per-role override, and None fallback."""

    def test_builds_with_default_model(self) -> None:
        """Builds model with default settings."""
        with patch("am_diag.agents.models.ChatOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.model = "gpt-4o"
            mock_cls.return_value = mock_instance

            settings = AgentSettings(api_key="test-key")
            model = build_chat_model("orchestrator", settings)
            assert model.model == "gpt-4o"
            mock_cls.assert_called_once()

    def test_per_role_override(self) -> None:
        """Per-role model override takes precedence."""
        with patch("am_diag.agents.models.ChatOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.model = "gpt-4o-mini"
            mock_cls.return_value = mock_instance

            settings = AgentSettings(
                api_key="test-key",
                model="gpt-4o",
                researcher_model="gpt-4o-mini",
            )
            model = build_chat_model("researcher", settings)
            assert model.model == "gpt-4o-mini"

    def test_default_settings_when_none(self) -> None:
        """Builds model when settings is None."""
        with patch("am_diag.agents.models.ChatOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            model = build_chat_model("orchestrator")
            assert model is not None
