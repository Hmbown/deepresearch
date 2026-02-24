import asyncio
import importlib
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from tests._fakes import FakeLLM, FakeSupervisorGraph


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _load_graph_module():
    graph = importlib.import_module("deepresearch.graph")
    return importlib.reload(graph)


def test_persisted_thread_checkpointer_clarify_then_proceed(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    report = importlib.import_module("deepresearch.report")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")

    llm = FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Which semiconductor segment should I focus on?",
                    verification="",
                ),
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="Understood. I will start focused research now.",
                ),
            ],
            "ResearchBrief": [SimpleNamespace(research_brief="Semiconductor datacenter GPU brief.")],
        },
        freeform_responses=[AIMessage(content="Final synthesized answer [1].")],
    )
    supervisor_graph = FakeSupervisorGraph(
        responses=[
            {
                "supervisor_messages": [HumanMessage(content="Semiconductor datacenter GPU brief.")],
                "notes": ["supervisor note [1]"],
                "raw_notes": ["supervisor raw [1] https://example.com/source-a"],
            }
        ]
    )

    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(report, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph.ainvoke)
    monkeypatch.setattr(graph, "build_supervisor_subgraph", lambda: supervisor_graph.ainvoke)

    app = graph.build_app(checkpointer=MemorySaver())
    cfg = _thread_config("thread-persisted-clarify")

    first = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="Tell me about semiconductors")]}, config=cfg))
    assert first["intake_decision"] == "clarify"
    assert first["awaiting_clarification"] is True
    assert "segment" in first["messages"][-1].content.lower()

    second = asyncio.run(
        app.ainvoke({"messages": [HumanMessage(content="Focus on high-end datacenter GPUs.")]}, config=cfg)
    )
    assert second["intake_decision"] == "proceed"
    assert second["awaiting_clarification"] is False
    assert "Final synthesized answer [1]." in second["final_report"]
    assert second["notes"] == ["supervisor note [1]"]
    assert second["raw_notes"] == ["supervisor raw [1] https://example.com/source-a"]
    assert supervisor_graph.calls == 1


def test_persisted_thread_topic_shift_resets_state_without_stale_note_leakage(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    report = importlib.import_module("deepresearch.report")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")

    llm = FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="I have enough context and will start now.",
                ),
                SimpleNamespace(
                    need_clarification=True,
                    question="Do you want me to switch fully to renewable energy supply chain?",
                    verification="",
                ),
            ],
            "ResearchBrief": [SimpleNamespace(research_brief="Generative AI healthcare adoption brief.")],
        },
        freeform_responses=[AIMessage(content="First run final report [1].")],
    )
    supervisor_graph = FakeSupervisorGraph(
        responses=[
            {
                "supervisor_messages": [HumanMessage(content="Generative AI healthcare adoption brief.")],
                "notes": ["existing note [1]"],
                "raw_notes": ["existing raw [1] https://example.com/source-old"],
            }
        ]
    )

    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(report, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph.ainvoke)
    monkeypatch.setattr(graph, "build_supervisor_subgraph", lambda: supervisor_graph.ainvoke)

    app = graph.build_app(checkpointer=MemorySaver())
    cfg = _thread_config("thread-persisted-shift")

    first = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="Research generative AI applications in healthcare.")]},
            config=cfg,
        )
    )
    assert first["notes"] == ["existing note [1]"]
    assert first["raw_notes"] == ["existing raw [1] https://example.com/source-old"]

    second = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="Switch to renewable energy supply chain instead.")]},
            config=cfg,
        )
    )
    assert second["intake_decision"] == "clarify"
    assert second["awaiting_clarification"] is True
    assert second["research_brief"] is None
    assert second["supervisor_messages"] == []
    assert second["notes"] == []
    assert second["raw_notes"] == []
    assert second["final_report"] == ""
    assert "switch fully" in second["messages"][-1].content.lower()
    assert supervisor_graph.calls == 1


def test_persisted_follow_up_continuity_resets_handoff_notes_before_new_supervisor_run(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    report = importlib.import_module("deepresearch.report")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")

    llm = FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="I have enough context and will start now.",
                )
            ],
            "ResearchBrief": [
                SimpleNamespace(research_brief="Battery recycling policy trends brief."),
                SimpleNamespace(research_brief="Battery recycling policy enforcement follow-up brief."),
            ],
        },
        freeform_responses=[
            AIMessage(content="First final answer [1]."),
            AIMessage(content="Second final answer [2]."),
        ],
    )
    supervisor_graph = FakeSupervisorGraph(
        responses=[
            {
                "supervisor_messages": [HumanMessage(content="Battery recycling policy trends brief.")],
                "notes": ["first note [1]"],
                "raw_notes": ["first raw [1] https://example.com/source-a"],
            },
            {
                "supervisor_messages": [HumanMessage(content="Battery recycling policy enforcement follow-up brief.")],
                "notes": ["second note [2]"],
                "raw_notes": ["second raw [2] https://example.com/source-b"],
            },
        ]
    )

    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(report, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph.ainvoke)
    monkeypatch.setattr(graph, "build_supervisor_subgraph", lambda: supervisor_graph.ainvoke)

    app = graph.build_app(checkpointer=MemorySaver())
    cfg = _thread_config("thread-persisted-continuity")

    first = asyncio.run(
        app.ainvoke({"messages": [HumanMessage(content="Research battery recycling policy trends.")]}, config=cfg)
    )
    assert first["notes"] == ["first note [1]"]
    assert first["raw_notes"] == ["first raw [1] https://example.com/source-a"]

    second = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="Add more detail on battery recycling policy enforcement timelines.")]},
            config=cfg,
        )
    )
    assert second["intake_decision"] == "proceed"
    assert second["awaiting_clarification"] is False
    assert second["notes"] == ["second note [2]"]
    assert second["raw_notes"] == ["second raw [2] https://example.com/source-b"]
    assert "first note [1]" not in second["notes"]
    assert "first raw [1]" not in second["raw_notes"][0]
    assert supervisor_graph.calls == 2


def test_persisted_thread_evidence_ledger_continuity(monkeypatch):
    """Evidence ledger fields survive checkpointed thread turns without duplication."""
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    report = importlib.import_module("deepresearch.report")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")

    llm = FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="Starting research now.",
                ),
            ],
            "ResearchBrief": [
                SimpleNamespace(research_brief="Evidence ledger test brief."),
            ],
        },
        freeform_responses=[AIMessage(content="Report with evidence [1][2].")],
    )
    supervisor_graph_mock = FakeSupervisorGraph(
        responses=[
            {
                "supervisor_messages": [HumanMessage(content="Evidence ledger test brief.")],
                "notes": ["note with evidence [1]"],
                "raw_notes": ["raw with evidence https://example.com/evidence-a"],
                "evidence_ledger": [
                    {
                        "source_urls": ["https://example.com/evidence-a"],
                    },
                    {
                        "source_urls": ["https://example.org/evidence-b"],
                    },
                ],
            }
        ]
    )

    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(report, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph_mock.ainvoke)
    monkeypatch.setattr(graph, "build_supervisor_subgraph", lambda: supervisor_graph_mock.ainvoke)

    app = graph.build_app(checkpointer=MemorySaver())
    cfg = _thread_config("thread-evidence-ledger")

    result = asyncio.run(
        app.ainvoke(
            {"messages": [HumanMessage(content="Research evidence-based topic.")]},
            config=cfg,
        )
    )

    from deepresearch.state import normalize_evidence_ledger

    normalized_evidence = normalize_evidence_ledger(result["evidence_ledger"])
    assert result["intake_decision"] == "proceed"
    assert normalized_evidence
    assert any(r.source_urls for r in normalized_evidence)
    assert "Sources:" in result["final_report"]


def _try_import_sqlite_saver():
    """Try to import SqliteSaver; skip test if unavailable."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver
    except ImportError:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            return AsyncSqliteSaver
        except ImportError:
            return None


def test_sqlite_checkpointer_clarify_then_proceed(monkeypatch, tmp_path):
    """Verify the clarify-then-proceed flow works with a real SQLite checkpointer."""
    SqliteSaver = _try_import_sqlite_saver()
    if SqliteSaver is None:
        pytest.skip("langgraph SQLite checkpointer not available")

    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    report = importlib.import_module("deepresearch.report")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")

    llm = FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Which area should I focus on?",
                    verification="",
                ),
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="Understood. Starting research.",
                ),
            ],
            "ResearchBrief": [SimpleNamespace(research_brief="SQLite test brief.")],
        },
        freeform_responses=[AIMessage(content="SQLite checkpointed answer [1].")],
    )
    supervisor_graph_mock = FakeSupervisorGraph(
        responses=[
            {
                "supervisor_messages": [HumanMessage(content="SQLite test brief.")],
                "notes": ["sqlite note [1]"],
                "raw_notes": ["sqlite raw [1] https://example.com/sqlite-source"],
            }
        ]
    )

    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(report, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph_mock.ainvoke)
    monkeypatch.setattr(graph, "build_supervisor_subgraph", lambda: supervisor_graph_mock.ainvoke)

    db_path = str(tmp_path / "test_checkpoint.db")

    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        checkpointer = SqliteSaver(conn)
        app = graph.build_app(checkpointer=checkpointer)
        cfg = _thread_config("thread-sqlite-test")

        first = asyncio.run(
            app.ainvoke({"messages": [HumanMessage(content="Research a broad topic.")]}, config=cfg)
        )
        assert first["intake_decision"] == "clarify"
        assert first["awaiting_clarification"] is True

        second = asyncio.run(
            app.ainvoke({"messages": [HumanMessage(content="Focus on a specific subtopic.")]}, config=cfg)
        )
        assert second["intake_decision"] == "proceed"
        assert "SQLite checkpointed answer [1]." in second["final_report"]
    finally:
        conn.close()
