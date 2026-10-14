"""Tests for the LangGraph-to-AgentEvent stream mapper.

``map_event`` translates LangGraph ``astream_events`` into the typed AgentEvent
union. Tool and token events map by event kind; structured planner/verifier/final
outputs map by domain type. Unrecognized events return ``None`` so the stream can
skip them. Synthetic event dicts exercise the mapping logic (the precise real
event shapes are a manual-verification step against a live trace).
"""

from __future__ import annotations

from am_diag.agents.events import map_event
from am_diag.common.data_models.agent import (
    AgentPlan,
    ClinicalAnswer,
    SubQuestion,
    SufficientContext,
)
from am_diag.common.data_models.agent_events import (
    FinalEvent,
    PlanEvent,
    TokenEvent,
    ToolCallEvent,
    VerifierEvent,
)


class TestMapEvent:
    """Tests for the LangGraph-to-AgentEvent mapping."""

    def test_tool_start_maps_to_tool_call(self):
        """A tool-start event becomes a ToolCallEvent with args."""
        ev = map_event(
            {
                "event": "on_tool_start",
                "name": "entity_search",
                "data": {"input": {"query": "sepsis"}},
            }
        )
        assert isinstance(ev, ToolCallEvent)
        assert ev.tool == "entity_search"
        assert ev.args == {"query": "sepsis"}

    def test_tool_end_maps_to_tool_call_preview(self):
        """A tool-end event carries a truncated result preview."""
        output = "0123456789" * 100
        ev = map_event(
            {"event": "on_tool_end", "name": "t", "data": {"output": output}}
        )
        assert isinstance(ev, ToolCallEvent)
        assert ev.result_preview is not None
        assert ev.result_preview == output[:200]

    def test_chat_stream_object_chunk_maps_to_token(self):
        """A model-stream chunk object becomes a TokenEvent."""
        chunk = type("Chunk", (), {"content": "hello"})()
        ev = map_event({"event": "on_chat_model_stream", "data": {"chunk": chunk}})
        assert isinstance(ev, TokenEvent)
        assert ev.text == "hello"

    def test_chat_stream_dict_chunk_maps_to_token(self):
        """A model-stream chunk dict is handled as well as an object."""
        ev = map_event(
            {"event": "on_chat_model_stream", "data": {"chunk": {"content": "hi"}}}
        )
        assert isinstance(ev, TokenEvent)
        assert ev.text == "hi"

    def test_list_content_chunk_maps_to_text_blocks_only(self):
        """Content-block chunks yield their text, not the list repr."""
        chunk = {
            "content": [
                {"type": "text", "text": "wor"},
                {"type": "tool_use", "id": "x"},
                "ds",
            ]
        }
        ev = map_event({"event": "on_chat_model_stream", "data": {"chunk": chunk}})
        assert isinstance(ev, TokenEvent)
        assert ev.text == "words"

    def test_subagent_token_is_skipped(self):
        """Tokens from a nested subagent graph are not user-facing."""
        ev = map_event(
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_checkpoint_ns": "tools:1|model:2"},
                "data": {"chunk": {"content": "planning"}},
            }
        )
        assert ev is None

    def test_orchestrator_token_is_kept(self):
        """Tokens from the top-level graph map to a TokenEvent."""
        ev = map_event(
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_checkpoint_ns": "model:1"},
                "data": {"chunk": {"content": "answer"}},
            }
        )
        assert isinstance(ev, TokenEvent)

    def test_empty_token_is_skipped(self):
        """An empty content chunk yields no event."""
        assert (
            map_event({"event": "on_chat_model_stream", "data": {"chunk": {}}}) is None
        )

    def test_typed_plan_output_maps_to_plan_event(self):
        """A typed AgentPlan output becomes a PlanEvent."""
        plan = AgentPlan(sub_questions=[SubQuestion(text="why?")])
        ev = map_event({"event": "on_chain_end", "data": {"output": plan}})
        assert isinstance(ev, PlanEvent)
        assert ev.plan is plan

    def test_typed_verifier_output_maps_to_verifier_event(self):
        """A typed SufficientContext output becomes a VerifierEvent."""
        assessment = SufficientContext(is_sufficient=True, coverage_score=3)
        ev = map_event({"event": "on_chain_end", "data": {"output": assessment}})
        assert isinstance(ev, VerifierEvent)
        assert ev.assessment is assessment

    def test_typed_final_answer_maps_to_final_event(self):
        """A typed ClinicalAnswer output becomes a FinalEvent."""
        answer = ClinicalAnswer(answer="lactate rises from anaerobic glycolysis")
        ev = map_event({"event": "on_chain_end", "data": {"output": answer}})
        assert isinstance(ev, FinalEvent)
        assert ev.answer is answer

    def test_structured_response_dict_maps_to_final_event(self):
        """A structured_response dict is coerced to the final ClinicalAnswer."""
        ev = map_event(
            {
                "event": "on_chain_end",
                "data": {"output": {"structured_response": {"answer": "yes"}}},
            }
        )
        assert isinstance(ev, FinalEvent)
        assert ev.answer.answer == "yes"

    def test_wrapped_plan_maps_to_plan_event(self):
        """A planner subagent's structured_response becomes a PlanEvent."""
        plan = AgentPlan(sub_questions=[SubQuestion(text="why?")])
        ev = map_event(
            {
                "event": "on_chain_end",
                "data": {"output": {"messages": [], "structured_response": plan}},
            }
        )
        assert isinstance(ev, PlanEvent)
        assert ev.plan is plan

    def test_wrapped_verifier_maps_to_verifier_event(self):
        """A verifier subagent's structured_response becomes a VerifierEvent."""
        assessment = SufficientContext(is_sufficient=True, coverage_score=3)
        ev = map_event(
            {
                "event": "on_chain_end",
                "data": {"output": {"structured_response": assessment}},
            }
        )
        assert isinstance(ev, VerifierEvent)
        assert ev.assessment is assessment

    def test_chain_end_without_structured_output_is_skipped(self):
        """A chain-end with an unrelated dict output yields no event."""
        ev = map_event({"event": "on_chain_end", "data": {"output": {"messages": []}}})
        assert ev is None

    def test_unknown_event_returns_none(self):
        """An unrecognized event is skipped."""
        assert map_event({"event": "on_chain_start", "data": {}}) is None
