import asyncio
import importlib

from langchain_core.messages import AIMessage
from langgraph.types import Send

from deepresearch import supervisor_subgraph
from deepresearch.state import FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH


def test_render_supervisor_prompt_excludes_downstream_prompt_contracts():
    rendered = supervisor_subgraph.render_supervisor_prompt(current_date="2026-02-24")

    assert "research supervisor" in rendered.lower()
    assert "ConductResearch" in rendered
    assert "Compression policy for supervisor notes:" not in rendered
    assert "Final report policy:" not in rendered


def test_supervisor_terminal_no_findings_uses_explicit_fallback(monkeypatch):
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    events = []
    monkeypatch.setattr(
        module,
        "log_runtime_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    result = asyncio.run(
        module.supervisor_terminal(
            {
                "supervisor_messages": [AIMessage(content="I can proceed without tools.")],
                "notes": [],
                "raw_notes": [],
                "evidence_ledger": [],
            }
        )
    )

    assert result["final_report"] == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["messages"][-1].content == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["notes"] == []
    assert result["raw_notes"] == []
    assert any(
        event == "supervisor_no_useful_research_fallback" and payload.get("reason") == "no_notes"
        for event, payload in events
    )


def test_supervisor_terminal_exception_path_uses_explicit_fallback(monkeypatch):
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    events = []
    monkeypatch.setattr(
        module,
        "log_runtime_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    result = asyncio.run(
        module.supervisor_terminal(
            {
                "supervisor_messages": [],
                "notes": [],
                "raw_notes": [],
                "evidence_ledger": [],
                "supervisor_exception": "boom",
            }
        )
    )

    assert result["final_report"] == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["messages"][-1].content == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert any(
        event == "supervisor_no_useful_research_fallback" and payload.get("reason") == "exception"
        for event, payload in events
    )


def test_supervisor_terminal_healthy_path_unchanged(monkeypatch):
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    events = []
    monkeypatch.setattr(
        module,
        "log_runtime_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    result = asyncio.run(
        module.supervisor_terminal(
            {
                "supervisor_messages": [
                    AIMessage(
                        content="delegating",
                        tool_calls=[{"id": "c1", "name": "ConductResearch", "args": {"research_topic": "x"}}],
                    )
                ],
                "notes": ["new note [1]"],
                "raw_notes": ["new raw [1]"],
            }
        )
    )

    assert "final_report" not in result
    assert "messages" not in result
    assert result["intake_decision"] == "proceed"
    assert result["awaiting_clarification"] is False
    assert not any(event == "supervisor_no_useful_research_fallback" for event, _ in events)


def test_route_supervisor_prepare_returns_send_dispatches():
    dispatch = supervisor_subgraph.route_supervisor_prepare(
        {
            "pending_research_calls": [
                {"id": "call-1", "args": {"research_topic": "Topic A"}, "topic": "Topic A"},
                {"id": "call-2", "args": {"research_topic": "Topic B"}, "topic": "Topic B"},
            ]
        }
    )

    assert isinstance(dispatch, list)
    assert len(dispatch) == 2
    assert all(isinstance(item, Send) for item in dispatch)
    assert dispatch[0].node == "run_research_unit"
