"""Tests for build_clinical_agent factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from am_diag.agents.agent import build_clinical_agent
from am_diag.agents.settings import AgentSettings
from am_diag.retrieval.search import SearchEngine


pytestmark = pytest.mark.enable_socket


class TestBuildClinicalAgent:
    async def test_returns_compiled_graph(self) -> None:
        with (
            patch("deepagents.create_deep_agent") as mock_factory,
            patch("am_diag.agents.agent.build_chat_model"),
            patch("am_diag.agents.subagents.build_chat_model"),
        ):
            mock_agent = MagicMock()
            mock_factory.return_value = mock_agent
            agent = await build_clinical_agent()
            assert agent is mock_agent
            mock_factory.assert_called_once()

    async def test_passes_search_engine_to_researcher(self) -> None:
        with (
            patch("deepagents.create_deep_agent") as mock_factory,
            patch("am_diag.agents.agent.build_chat_model"),
            patch("am_diag.agents.subagents.build_chat_model"),
            patch("am_diag.agents.agent.build_researcher_subagent") as mock_build_res,
        ):
            mock_factory.return_value = MagicMock()

            engine = MagicMock(spec=SearchEngine)
            await build_clinical_agent(search_engine=engine)

            mock_build_res.assert_called_once_with(engine, AgentSettings())

    async def test_uses_custom_settings(self) -> None:
        settings = AgentSettings(model="gpt-5", api_key="test")
        with (
            patch("deepagents.create_deep_agent") as mock_factory,
            patch("am_diag.agents.agent.build_chat_model"),
            patch("am_diag.agents.subagents.build_chat_model"),
        ):
            mock_factory.return_value = MagicMock()

            agent = await build_clinical_agent(settings=settings)
            assert agent is not None

    async def test_agent_has_response_format(self) -> None:
        with (
            patch("deepagents.create_deep_agent") as mock_factory,
            patch("am_diag.agents.agent.build_chat_model"),
            patch("am_diag.agents.subagents.build_chat_model"),
        ):
            mock_factory.return_value = MagicMock()

            await build_clinical_agent()
            call_kwargs = mock_factory.call_args.kwargs
            assert "response_format" in call_kwargs
            assert "ClinicalAnswer" in str(call_kwargs["response_format"])

    async def test_orchestrator_prompt_present(self) -> None:
        with (
            patch("deepagents.create_deep_agent") as mock_factory,
            patch("am_diag.agents.agent.build_chat_model"),
            patch("am_diag.agents.subagents.build_chat_model"),
        ):
            mock_factory.return_value = MagicMock()

            await build_clinical_agent()
            call_kwargs = mock_factory.call_args.kwargs
            assert "system_prompt" in call_kwargs
            assert "answerable" in call_kwargs["system_prompt"].lower()
