"""Tests for the project-scoped agent answer surface.

The DeepAgents graph is replaced with a fake agent (mocked ainvoke), so answer()
is verified for both string and multi-turn message inputs without building the
real agent or touching any backend.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from am_diag.api.project import Project
from am_diag.api.registry import Registry
from am_diag.api.workspace import Workspace
from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.connection import GraphConnection, VectorConnection
from am_diag.common.data_models.graph_record import GraphRecord
from am_diag.common.data_models.message import Message
from am_diag.common.data_models.project import ProjectRecord


def _setup(tmp_path, monkeypatch, agent_out):
    """Register a project; patch Project._agent to return a fake agent."""
    registry = Registry.open(tmp_path / "registry.json")
    client = Workspace(registry, state_dir=tmp_path / "state")
    registry.add_graph(
        GraphRecord(
            graph_id="med",
            neo4j=GraphConnection(uri="bolt://h", user_env="U", password_env="P"),
            vector=VectorConnection(url_env="URL"),
        )
    )
    registry.add_project(ProjectRecord(project_id="p", graph_ids=["med"]))
    fake_agent = SimpleNamespace(ainvoke=AsyncMock(return_value=agent_out))

    async def fake_build(self, graphs, settings):
        return fake_agent

    monkeypatch.setattr(Project, "_agent", fake_build)
    return client, fake_agent


class TestAgentAnswer:
    """Tests for Project.answer with string and message inputs."""

    async def test_answer_string_returns_clinical_answer(self, tmp_path, monkeypatch):
        """Verify a string question yields a ClinicalAnswer."""
        client, _ = _setup(
            tmp_path, monkeypatch, {"structured_response": ClinicalAnswer(answer="x")}
        )
        ans = await client.project("p").answer("Why is lactate elevated?")
        assert isinstance(ans, ClinicalAnswer)
        assert ans.answer == "x"

    async def test_answer_messages_accepted(self, tmp_path, monkeypatch):
        """Verify a list of Message turns is accepted and forwarded."""
        client, agent = _setup(
            tmp_path, monkeypatch, {"structured_response": ClinicalAnswer(answer="y")}
        )
        ans = await client.project("p").answer(
            [Message(role="user", content="hi"), Message(role="assistant", content="?")]
        )
        assert ans.answer == "y"
        sent = agent.ainvoke.call_args.args[0]["messages"]
        assert sent == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "?"},
        ]
