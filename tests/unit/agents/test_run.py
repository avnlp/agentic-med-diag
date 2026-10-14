"""Tests for answer_question convenience entrypoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from am_diag.agents.run import answer_question, stream_answer
from am_diag.common.data_models.agent import ClinicalAnswer
from am_diag.common.data_models.agent_events import FinalEvent, TokenEvent


class TestAnswerQuestion:
    async def test_returns_structured_response(self) -> None:
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "structured_response": {"answer": "test", "answerable": True},
            },
        )

        with patch(
            "am_diag.agents.run.build_clinical_agent",
            return_value=mock_agent,
        ):
            result = await answer_question("What is diabetes?")

        assert result["answer"] == "test"
        assert result["answerable"] is True

    async def test_fallback_from_messages(self) -> None:
        class _FakeMessage:
            content: str = "fallback answer"

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [_FakeMessage()],
                "structured_response": None,
            },
        )

        with patch(
            "am_diag.agents.run.build_clinical_agent",
            return_value=mock_agent,
        ):
            result = await answer_question("What is diabetes?")

        assert isinstance(result, object)

    async def test_no_structured_response_raises(self) -> None:
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={})

        with (
            patch(
                "am_diag.agents.run.build_clinical_agent",
                return_value=mock_agent,
            ),
            pytest.raises(RuntimeError, match="Unexpected agent output structure"),
        ):
            await answer_question("What is diabetes?")

    async def test_passes_search_engine(self) -> None:
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "structured_response": {"answer": "ok", "answerable": True},
            },
        )

        with patch(
            "am_diag.agents.run.build_clinical_agent",
            return_value=mock_agent,
        ) as mock_build:
            engine = MagicMock()
            await answer_question("Q?", search_engine=engine)
            mock_build.assert_called_once()
            assert "search_engine" in mock_build.call_args.kwargs


class TestStreamAnswer:
    async def test_yields_mapped_events_and_skips_unmapped(self) -> None:
        async def _astream_events(*args, **kwargs):
            yield {"event": "on_chain_start", "data": {}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": {"content": "a"}}}
            yield {
                "event": "on_chain_end",
                "data": {"output": ClinicalAnswer(answer="done")},
            }

        mock_agent = MagicMock()
        mock_agent.astream_events = _astream_events

        with patch(
            "am_diag.agents.run.build_clinical_agent",
            return_value=mock_agent,
        ):
            events = [event async for event in stream_answer("Q?")]

        assert [type(event) for event in events] == [TokenEvent, FinalEvent]
