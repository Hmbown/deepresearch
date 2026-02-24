import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage

from deepresearch import supervisor_subgraph
from deepresearch.state import FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH


def _load_graph_module():
    graph = importlib.import_module("deepresearch.graph")
    return importlib.reload(graph)


def test_render_supervisor_prompt_excludes_downstream_prompt_contracts():
    rendered = supervisor_subgraph.render_supervisor_prompt(current_date="2026-02-24")

    assert "Research Supervisor orchestrating deep research" in rendered
    assert "Compression policy for supervisor notes:" not in rendered
    assert "Final report policy:" not in rendered


def test_research_supervisor_no_tool_narrative_path_uses_explicit_fallback(monkeypatch):
    graph = _load_graph_module()
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    events = []

    supervisor_graph = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "supervisor_messages": [
                    HumanMessage(content="Scoped brief"),
                    AIMessage(content="I can proceed without tools."),
                ],
                "notes": [],
                "raw_notes": [],
            }
        )
    )

    monkeypatch.setattr(module, "build_supervisor_subgraph", lambda: supervisor_graph)
    monkeypatch.setattr(
        module,
        "log_runtime_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    result = asyncio.run(
        graph.research_supervisor(
            {
                "messages": [HumanMessage(content="Research topic")],
                "research_brief": "Scoped brief",
                "supervisor_messages": [],
                "notes": [],
                "raw_notes": [],
            }
        )
    )

    assert result["final_report"] == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["messages"][-1].content == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["notes"] == []
    assert result["raw_notes"] == []
    assert ("supervisor_no_useful_research_fallback", {"reason": "no_notes", "had_tool_calls": False, "supervisor_message_count": 2, "error": None}) in events


def test_research_supervisor_empty_subgraph_output_uses_explicit_fallback(monkeypatch):
    graph = _load_graph_module()
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    events = []

    supervisor_graph = SimpleNamespace(ainvoke=AsyncMock(return_value={}))

    monkeypatch.setattr(module, "build_supervisor_subgraph", lambda: supervisor_graph)
    monkeypatch.setattr(
        module,
        "log_runtime_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    result = asyncio.run(
        graph.research_supervisor(
            {
                "messages": [HumanMessage(content="Research topic")],
                "research_brief": "Scoped brief",
                "supervisor_messages": [],
                "notes": [],
                "raw_notes": [],
            }
        )
    )

    assert result["final_report"] == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["messages"][-1].content == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["notes"] == []
    assert result["raw_notes"] == []
    assert ("supervisor_no_useful_research_fallback", {"reason": "no_notes", "had_tool_calls": False, "supervisor_message_count": 0, "error": None}) in events


def test_research_supervisor_exception_path_uses_explicit_fallback(monkeypatch):
    graph = _load_graph_module()
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    events = []

    supervisor_graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("boom")))

    monkeypatch.setattr(module, "build_supervisor_subgraph", lambda: supervisor_graph)
    monkeypatch.setattr(
        module,
        "log_runtime_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    result = asyncio.run(
        graph.research_supervisor(
            {
                "messages": [HumanMessage(content="Research topic")],
                "research_brief": "Scoped brief",
                "supervisor_messages": [],
                "notes": [],
                "raw_notes": [],
            }
        )
    )

    assert result["final_report"] == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["messages"][-1].content == FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
    assert result["notes"] == []
    assert result["raw_notes"] == []
    assert any(
        event == "supervisor_no_useful_research_fallback" and payload.get("reason") == "exception"
        for event, payload in events
    )


def test_research_supervisor_healthy_path_unchanged(monkeypatch):
    graph = _load_graph_module()
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    events = []

    supervisor_graph = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "supervisor_messages": [
                    HumanMessage(content="Scoped brief"),
                    AIMessage(content="delegating", tool_calls=[{"id": "c1", "name": "ConductResearch", "args": {"research_topic": "x"}}]),
                ],
                "notes": ["new note [1]"],
                "raw_notes": ["new raw [1]"],
            }
        )
    )

    monkeypatch.setattr(module, "build_supervisor_subgraph", lambda: supervisor_graph)
    monkeypatch.setattr(
        module,
        "log_runtime_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    result = asyncio.run(
        graph.research_supervisor(
            {
                "messages": [HumanMessage(content="Research topic")],
                "research_brief": "Scoped brief",
                "supervisor_messages": [],
                "notes": [],
                "raw_notes": [],
            }
        )
    )

    assert "final_report" not in result
    assert "messages" not in result
    assert result["notes"] == ["new note [1]"]
    assert result["raw_notes"] == ["new raw [1]"]
    assert not any(event == "supervisor_no_useful_research_fallback" for event, _ in events)
