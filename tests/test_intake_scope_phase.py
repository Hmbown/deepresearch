import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage


class _FakeStructuredRunner:
    def __init__(self, owner, schema_name):
        self._owner = owner
        self._schema_name = schema_name

    async def ainvoke(self, messages, config=None):
        del config
        self._owner.structured_calls.append((self._schema_name, messages))
        responses = self._owner.structured_responses.get(self._schema_name, [])
        if not responses:
            raise AssertionError(f"No fake response configured for schema {self._schema_name}")
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeLLM:
    def __init__(self, *, structured_responses=None, freeform_responses=None):
        self.structured_responses = {
            key: list(value) for key, value in (structured_responses or {}).items()
        }
        self.freeform_responses = list(freeform_responses or [])
        self.structured_calls = []
        self.freeform_calls = []

    def with_structured_output(self, schema):
        return _FakeStructuredRunner(self, schema.__name__)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, config=None):
        del config
        self.freeform_calls.append(messages)
        if not self.freeform_responses:
            return AIMessage(content="fallback freeform output")
        response = self.freeform_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _load_graph_module():
    graph = importlib.import_module("deepresearch.graph")
    graph = importlib.reload(graph)
    return graph


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def test_scope_intake_runs_intake_on_first_turn(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Which battery segment should I focus on?",
                    verification="",
                )
            ]
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [HumanMessage(content="What is new in battery research?")],
                "awaiting_clarification": False,
                "intake_decision": None,
                "research_brief": None,
            }
        )
    )

    assert command.goto == "__end__"
    assert command.update["intake_decision"] == "clarify"
    assert command.update["awaiting_clarification"] is True
    assert len(command.update["messages"]) == 1
    assert "battery segment" in command.update["messages"][0].content.lower()
    assert llm.structured_calls


def test_scope_intake_blocks_broad_request_without_boundary_before_model_call(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="I will start now.",
                )
            ]
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [HumanMessage(content="Which stocks are most likely to go bankrupt?")],
                "awaiting_clarification": False,
                "intake_decision": None,
                "research_brief": None,
            }
        )
    )

    assert command.goto == "__end__"
    assert command.update["intake_decision"] == "clarify"
    assert command.update["awaiting_clarification"] is True
    question = command.update["messages"][0].content.lower()
    assert "timeframe" in question
    assert "geography" in question or "universe" in question
    assert llm.structured_calls == []


def test_scope_intake_offers_plan_checkpoint_for_broad_request_with_scope(monkeypatch):
    """ClarifyWithUser runs first; when it says proceed on a broad query, a plan is shown."""
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="Understood, proceeding.",
                )
            ],
            "ResearchBrief": [
                SimpleNamespace(research_brief="U.S. bankruptcy-risk stock screen from Feb 2026 onward.")
            ],
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Which U.S. stocks are most at risk of bankruptcy from February 2026 onward?"
                        )
                    )
                ],
                "awaiting_clarification": False,
                "intake_decision": None,
                "research_brief": None,
            }
        )
    )

    assert command.goto == "__end__"
    assert command.update["intake_decision"] == "clarify"
    assert command.update["awaiting_clarification"] is True
    assert "reply \"start\"" in command.update["messages"][0].content.lower()
    schemas_called = [schema for schema, _ in llm.structured_calls]
    assert "ClarifyWithUser" in schemas_called
    assert "ResearchBrief" in schemas_called


def test_scope_intake_proceeds_after_plan_checkpoint_confirmation(monkeypatch):
    """When research_brief is set and awaiting_clarification is True, user response
    triggers fast path — regenerate brief from full conversation and proceed."""
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ResearchBrief": [
                SimpleNamespace(research_brief="U.S. bankruptcy-risk stock screen from Feb 2026 onward.")
            ],
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Which U.S. stocks are most at risk of bankruptcy from February 2026 onward?"
                        )
                    ),
                    AIMessage(content='If this plan looks right, reply "start".'),
                    HumanMessage(content="start"),
                ],
                "awaiting_clarification": True,
                "intake_decision": "clarify",
                "research_brief": "U.S. bankruptcy-risk stock screen from Feb 2026 onward.",
            }
        )
    )

    assert command.goto == "research_supervisor"
    assert command.update["intake_decision"] == "proceed"
    assert command.update["awaiting_clarification"] is False
    assert command.update["research_brief"]
    # Fast path should NOT run ClarifyWithUser — only ResearchBrief for regeneration
    schemas_called = [schema for schema, _ in llm.structured_calls]
    assert "ClarifyWithUser" not in schemas_called
    assert "ResearchBrief" in schemas_called


def test_scope_intake_fast_path_requires_clarify_state(monkeypatch):
    """Avoid fast-path regression when stale state sets `awaiting_clarification`.

    If intake_decision is not "clarify", a stale research brief from prior runs must
    not be treated as a plan-acknowledgment path.
    """
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Do you want to switch fully to the new topic?",
                    verification="",
                )
            ],
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(content="Research generative AI applications"),
                    AIMessage(content="I can start with the prior brief."),
                    HumanMessage(content="Switch to renewable energy instead."),
                ],
                "awaiting_clarification": True,
                "intake_decision": "proceed",
                "research_brief": "Prior research brief to be replaced",
            }
        )
    )

    assert command.goto == "__end__"
    assert command.update["intake_decision"] == "clarify"
    assert command.update["awaiting_clarification"] is True
    assert command.update["research_brief"] is None
    assert [schema for schema, _ in llm.structured_calls] == ["ClarifyWithUser"]


def test_scope_intake_bypasses_clarify_after_proceed(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ResearchBrief": [SimpleNamespace(research_brief="Follow-up brief for supervisor.")],
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(content="Research generative AI applications"),
                    HumanMessage(content="Add more detail on enterprise adoption"),
                ],
                "awaiting_clarification": False,
                "intake_decision": "proceed",
                "research_brief": "Scoped AI applications brief",
            }
        )
    )

    assert command.goto == "research_supervisor"
    assert command.update["research_brief"] == "Follow-up brief for supervisor."
    assert command.update["intake_decision"] == "proceed"
    assert command.update["awaiting_clarification"] is False
    assert "messages" not in command.update
    assert [schema for schema, _ in llm.structured_calls] == ["ResearchBrief"]


def test_scope_intake_rechecks_intent_for_topic_shift_follow_up(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Do you want me to switch fully to renewable energy supply chain?",
                    verification="",
                )
            ]
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(content="Research generative AI applications"),
                    HumanMessage(content="Switch to renewable energy supply chain instead."),
                ],
                "awaiting_clarification": False,
                "intake_decision": "proceed",
                "research_brief": "Scoped AI applications brief",
            }
        )
    )

    assert command.goto == "__end__"
    assert [schema for schema, _ in llm.structured_calls] == ["ClarifyWithUser"]


def test_scope_intake_topic_shift_clarify_hard_resets_prior_state(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Do you want me to switch fully to renewable energy supply chain?",
                    verification="",
                )
            ]
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(content="Research generative AI applications"),
                    HumanMessage(content="Switch to renewable energy supply chain instead."),
                ],
                "awaiting_clarification": False,
                "intake_decision": "proceed",
                "research_brief": "Scoped AI applications brief",
                "supervisor_messages": [AIMessage(content="Prior supervisor state")],
                "notes": ["existing note [1]"],
                "raw_notes": ["existing raw [1]"],
                "final_report": "Old report [1]",
            }
        )
    )

    assert command.goto == "__end__"
    assert command.update["intake_decision"] == "clarify"
    assert command.update["awaiting_clarification"] is True
    assert command.update["research_brief"] is None
    assert command.update["supervisor_messages"] == []
    assert command.update["notes"] == []
    assert command.update["raw_notes"] == []
    assert command.update["final_report"] == ""


def test_scope_intake_keeps_same_topic_follow_up_with_new_as_proceed(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Should I switch topics?",
                    verification="",
                )
            ],
            "ResearchBrief": [SimpleNamespace(research_brief="Battery recycling policy follow-up brief.")],
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(content="Research battery recycling policy trends"),
                    HumanMessage(content="What new battery recycling policies were announced in 2025?"),
                ],
                "awaiting_clarification": False,
                "intake_decision": "proceed",
                "research_brief": "Battery recycling policy brief",
            }
        )
    )

    assert command.goto == "research_supervisor"
    assert [schema for schema, _ in llm.structured_calls] == ["ResearchBrief"]


def test_app_stops_at_clarification_when_needed(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Which market segment do you want to focus on?",
                    verification="",
                )
            ]
        }
    )
    supervisor_graph = SimpleNamespace(ainvoke=AsyncMock(return_value={}))

    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph)

    result = asyncio.run(
        graph.app.ainvoke(
            {"messages": [HumanMessage(content="Tell me about semiconductors")]},
            config=_thread_config("thread-clarify"),
        )
    )

    assert result["intake_decision"] == "clarify"
    assert result["awaiting_clarification"] is True
    assert "market segment" in result["messages"][-1].content.lower()
    assert supervisor_graph.ainvoke.await_count == 0


def test_app_multi_turn_clarify_then_proceed_uses_message_history(monkeypatch):
    intake = importlib.import_module("deepresearch.intake")
    report = importlib.import_module("deepresearch.report")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=True,
                    question="Which market segment do you want to focus on?",
                    verification="",
                ),
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="Understood. I will start research now.",
                ),
            ],
            "ResearchBrief": [SimpleNamespace(research_brief="Research brief for test.")],
        },
        freeform_responses=[AIMessage(content="Final synthesized answer [1].")],
    )
    supervisor_graph = AsyncMock(
        return_value={
            "supervisor_messages": [HumanMessage(content="Research brief for test.")],
            "notes": ["supervisor note [1]"],
            "raw_notes": ["raw note [1]"],
        }
    )

    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(report, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph)
    graph = _load_graph_module()

    first_result = asyncio.run(
        graph.app.ainvoke(
            {"messages": [HumanMessage(content="Tell me about semiconductors")]},
            config=_thread_config("thread-clarify-proceed"),
        )
    )

    assert first_result["intake_decision"] == "clarify"
    assert first_result["awaiting_clarification"] is True
    assert "market segment" in first_result["messages"][-1].content.lower()
    assert supervisor_graph.await_count == 0

    follow_up_messages = list(first_result["messages"]) + [
        HumanMessage(content="Focus on high-end datacenter GPUs.")
    ]
    second_result = asyncio.run(
        graph.app.ainvoke(
            {"messages": follow_up_messages},
            config=_thread_config("thread-clarify-proceed"),
        )
    )

    assert second_result["intake_decision"] == "proceed"
    assert second_result["awaiting_clarification"] is False
    assert "Final synthesized answer [1]." in second_result["final_report"]
    assert "Sources:" in second_result["final_report"]
    assert supervisor_graph.await_count == 1


def test_scope_intake_initializes_supervisor_state(monkeypatch):
    graph = _load_graph_module()
    intake = importlib.import_module("deepresearch.intake")
    llm = _FakeLLM(
        structured_responses={
            "ResearchBrief": [SimpleNamespace(research_brief="Detailed brief for supervision.")]
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)

    command = asyncio.run(
        graph.scope_intake(
            {
                "messages": [
                    HumanMessage(content="Research enterprise software adoption"),
                    HumanMessage(content="Add more details on enterprise adoption"),
                ],
                "awaiting_clarification": False,
                "intake_decision": "proceed",
                "research_brief": "Prior brief",
            }
        )
    )
    result = command.update

    assert command.goto == "research_supervisor"
    assert result["research_brief"] == "Detailed brief for supervision."
    assert len(result["supervisor_messages"]) == 1
    assert result["supervisor_messages"][0].type == "human"
    assert result["notes"] == []
    assert result["raw_notes"] == []
    assert result["final_report"] == ""
    assert result["intake_decision"] == "proceed"
    assert result["awaiting_clarification"] is False


def test_supervisor_prepare_runs_parallel_dispatch_planning_and_enforces_cap(monkeypatch):
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    monkeypatch.setattr(supervisor_subgraph, "get_max_concurrent_research_units", lambda: 1)
    monkeypatch.setattr(supervisor_subgraph, "get_max_researcher_iterations", lambda: 3)

    latest_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "name": "ConductResearch",
                "args": {"research_topic": "Topic A"},
            },
            {
                "id": "call-2",
                "name": "ConductResearch",
                "args": {"research_topic": "Topic B"},
            },
        ],
    )
    state = {
        "supervisor_messages": [latest_ai],
        "research_iterations": 0,
    }

    result = asyncio.run(supervisor_subgraph.supervisor_prepare(state))

    assert result["pending_requested_research_units"] == 2
    assert result["pending_dispatched_research_units"] == 1
    assert result["pending_skipped_research_units"] == 1
    assert len(result["pending_research_calls"]) == 1
    assert any("skipped" in message.content for message in result["supervisor_messages"])


def test_supervisor_run_research_unit_extracts_notes_and_evidence(monkeypatch):
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    researcher_graph = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "messages": [
                    AIMessage(content="compressed finding [1]\\n\\nSources:\\n[1] https://example.com/source"),
                ]
            }
        )
    )
    monkeypatch.setattr(supervisor_subgraph, "build_researcher_subgraph", lambda: researcher_graph)

    result = asyncio.run(
        supervisor_subgraph.run_research_unit(
            {"research_call": {"id": "call-1", "args": {"research_topic": "Topic A"}, "topic": "Topic A"}}
        )
    )

    assert result["notes"]
    assert any("finding [1]" in note for note in result["notes"])
    assert result["evidence_ledger"]
    assert researcher_graph.ainvoke.await_count == 1


def test_supervisor_finalize_marks_completion_when_research_complete_called(monkeypatch):
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    monkeypatch.setattr(supervisor_subgraph, "get_max_researcher_iterations", lambda: 6)

    state = {
        "pending_complete_calls": [{"id": "complete-1"}],
        "pending_requested_research_units": 0,
        "pending_dispatched_research_units": 0,
        "pending_skipped_research_units": 0,
        "pending_remaining_iterations": 4,
        "research_unit_summaries": [],
        "research_unit_summaries_consumed": 0,
        "evidence_ledger": [],
        "research_iterations": 2,
    }

    result = asyncio.run(supervisor_subgraph.supervisor_finalize(state))
    assert result["research_iterations"] == 6
    assert any("ResearchComplete received" in msg.content for msg in result["supervisor_messages"])


def test_final_report_generation_uses_model_output(monkeypatch):
    graph = _load_graph_module()
    report = importlib.import_module("deepresearch.report")
    llm = _FakeLLM(freeform_responses=[AIMessage(content="Final report with citations [1].")])
    monkeypatch.setattr(report, "get_llm", lambda role: llm)

    result = asyncio.run(
        graph.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": ["Finding [1]"],
                "raw_notes": ["Raw [1]"],
                "final_report": "",
            }
        )
    )

    assert "Final report with citations [1]." in result["final_report"]
    assert "Sources:" in result["final_report"]
    assert result["messages"][-1].content == result["final_report"]


def test_final_report_generation_retries_on_token_limit_then_succeeds(monkeypatch):
    graph = _load_graph_module()
    report = importlib.import_module("deepresearch.report")
    llm = _FakeLLM(
        freeform_responses=[
            RuntimeError("context length exceeded"),
            AIMessage(content="Recovered report"),
        ]
    )
    monkeypatch.setattr(report, "get_llm", lambda role: llm)

    result = asyncio.run(
        graph.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": ["Finding A [1]", "Finding B [2]", "Finding C [3]"],
                "raw_notes": ["Raw A", "Raw B", "Raw C"],
                "final_report": "",
            }
        )
    )

    assert "Recovered report" in result["final_report"]
    assert "Sources:" in result["final_report"]
    assert len(llm.freeform_calls) == 2


def test_app_proceed_flow_runs_supervisor_and_final_report(monkeypatch):
    intake = importlib.import_module("deepresearch.intake")
    report = importlib.import_module("deepresearch.report")
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    llm = _FakeLLM(
        structured_responses={
            "ClarifyWithUser": [
                SimpleNamespace(
                    need_clarification=False,
                    question="",
                    verification="I have enough context and will start now.",
                )
            ],
            "ResearchBrief": [SimpleNamespace(research_brief="Research brief for test.")],
        },
        freeform_responses=[AIMessage(content="Final synthesized answer [1].")],
    )
    supervisor_graph = AsyncMock(
        return_value={
            "supervisor_messages": [HumanMessage(content="Research brief for test.")],
            "notes": ["supervisor note [1]"],
            "raw_notes": ["raw note [1]"],
        }
    )
    monkeypatch.setattr(intake, "get_llm", lambda role: llm)
    monkeypatch.setattr(report, "get_llm", lambda role: llm)
    monkeypatch.setattr(supervisor_subgraph, "build_supervisor_subgraph", lambda: supervisor_graph)
    graph = _load_graph_module()

    result = asyncio.run(
        graph.app.ainvoke(
            {"messages": [HumanMessage(content="Explain retrieval-augmented generation limits")]},
            config=_thread_config("thread-proceed"),
        )
    )

    assert result["intake_decision"] == "proceed"
    assert result["awaiting_clarification"] is False
    assert "Final synthesized answer [1]." in result["final_report"]
    assert "Sources:" in result["final_report"]
    assert supervisor_graph.await_count == 1


def test_research_handoff_update_resets_accumulated_supervisor_and_note_state():
    from langgraph.graph import END, START, StateGraph

    from deepresearch.intake import _build_research_handoff_update
    from deepresearch.state import ResearchState

    def _handoff_node(_state):
        return _build_research_handoff_update("Fresh research brief")

    builder = StateGraph(ResearchState)
    builder.add_node("handoff", _handoff_node)
    builder.add_edge(START, "handoff")
    builder.add_edge("handoff", END)
    handoff_graph = builder.compile()

    result = asyncio.run(
        handoff_graph.ainvoke(
            {
                "messages": [HumanMessage(content="New query")],
                "supervisor_messages": [AIMessage(content="stale supervisor state")],
                "notes": ["stale note [1] https://stale.example"],
                "raw_notes": ["stale raw note [1] https://stale.example"],
            }
        )
    )

    assert result["notes"] == []
    assert result["raw_notes"] == []
    assert len(result["supervisor_messages"]) == 1
    assert result["supervisor_messages"][0].type == "human"
