import asyncio
import importlib

from langchain_core.messages import AIMessage
from langgraph.types import Send

from deepresearch import supervisor_subgraph
from deepresearch.state import FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH, today_utc_date

TEST_DATE = today_utc_date()


def test_render_supervisor_prompt_excludes_downstream_prompt_contracts():
    rendered = supervisor_subgraph.render_supervisor_prompt(current_date=TEST_DATE)

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


def test_research_barrier_waits_for_remaining_research_units():
    command = supervisor_subgraph.research_barrier(
        {
            "pending_dispatched_research_units": 2,
            "research_unit_summaries": [{"call_id": "call-1"}],
            "research_unit_summaries_consumed": 0,
        }
    )

    assert command.goto == ()


def test_research_barrier_routes_to_finalize_when_wave_is_complete():
    command = supervisor_subgraph.research_barrier(
        {
            "pending_dispatched_research_units": 2,
            "research_unit_summaries": [{"call_id": "call-1"}, {"call_id": "call-2"}],
            "research_unit_summaries_consumed": 0,
        }
    )

    assert command.goto == "supervisor_finalize"


def test_supervisor_finalize_progress_counts_only_fetched_evidence(monkeypatch):
    module = importlib.import_module("deepresearch.supervisor_subgraph")
    captured_payload: dict[str, object] = {}

    async def fake_dispatch(event_name, payload, config=None):
        del config
        captured_payload["event_name"] = event_name
        captured_payload["payload"] = payload

    monkeypatch.setattr(module, "adispatch_custom_event", fake_dispatch)
    monkeypatch.setattr(module, "get_max_researcher_iterations", lambda: 6)

    state = {
        "pending_complete_calls": [{"id": "complete-1"}],
        "pending_requested_research_units": 0,
        "pending_dispatched_research_units": 0,
        "pending_skipped_research_units": 0,
        "pending_remaining_iterations": 4,
        "pending_research_calls": [],
        "research_unit_summaries": [],
        "research_unit_summaries_consumed": 0,
        "research_iterations": 2,
        "evidence_ledger": [
            {"source_urls": ["https://example.com/a"], "source_type": "model_cited"},
            {"source_urls": ["https://example.com/a"], "source_type": "fetched"},
            {"source_urls": ["https://example.org/b"], "source_type": "model_cited"},
        ],
    }

    asyncio.run(module.supervisor_finalize(state))
    payload = captured_payload["payload"]
    assert captured_payload["event_name"] == "supervisor_progress"
    assert payload["evidence_record_count"] == 1
    assert payload["source_domains"] == ["example.com"]
    assert payload["model_cited_record_count"] == 1
    assert payload["model_cited_domains"] == ["example.org"]
