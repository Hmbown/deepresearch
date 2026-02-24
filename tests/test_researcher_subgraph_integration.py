import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepresearch import researcher_subgraph
from deepresearch.researcher_subgraph import extract_research_from_messages


class _FakeDeepResearcherGraph:
    """Simulates a deep agent compiled graph that processes MessagesState."""

    def __init__(self):
        self.calls = 0
        self.invoked_messages = None

    async def ainvoke(self, payload, *, config=None):
        self.invoked_messages = payload.get("messages", [])
        self.calls += 1
        # Simulate a deep agent that did search + fetch + synthesis
        return {
            "messages": [
                HumanMessage(content="Research integration topic"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "search-1",
                            "name": "search_web",
                            "args": {"query": "deterministic integration test query"},
                        }
                    ],
                ),
                ToolMessage(
                    content=(
                        "[Source 1] Example Source A\n"
                        "URL: https://example.com/source-a\n"
                        "Summary: Baseline measurement [1]."
                    ),
                    name="search_web",
                    tool_call_id="search-1",
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "fetch-1",
                            "name": "fetch_url",
                            "args": {"url": "https://example.com/source-a"},
                        }
                    ],
                ),
                ToolMessage(
                    content="Fetched detail [2] from https://example.com/source-b",
                    name="fetch_url",
                    tool_call_id="fetch-1",
                ),
                AIMessage(
                    content=(
                        "Executive Summary\n"
                        "Key claims are supported by corroborated evidence [1][2].\n\n"
                        "Evidence Log\n"
                        "- Source A supports baseline metrics [1].\n"
                        "- Source B confirms implementation detail [2].\n\n"
                        "Sources:\n"
                        "[1] https://example.com/source-a\n"
                        "[2] https://example.com/source-b"
                    )
                ),
            ]
        }


def test_researcher_deep_agent_loop_preserves_sources(monkeypatch):
    """Verify extract_research_from_messages extracts notes and citations from MessagesState."""
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    fake_graph = _FakeDeepResearcherGraph()

    monkeypatch.setattr(
        researcher_subgraph,
        "create_deep_agent",
        lambda **kwargs: fake_graph,
    )

    graph = researcher_subgraph.build_researcher_subgraph()
    result = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="Research integration topic")]},
        )
    )

    assert fake_graph.calls == 1
    assert fake_graph.invoked_messages is not None

    compressed, raw_notes, evidence_ledger = extract_research_from_messages(result)

    assert raw_notes
    assert any("https://example.com/source-a" in note for note in raw_notes)
    assert any("[1]" in note or "[2]" in note for note in raw_notes)
    assert evidence_ledger
    urls = {url for record in evidence_ledger for url in record.source_urls}
    assert urls == {
        "https://example.com/source-a",
        "https://example.com/source-b",
    }
    assert compressed is not None
    assert "[1]" in compressed or "https://example.com/source-a" in compressed


def test_extract_research_from_messages_excludes_think_tool():
    """Verify think_tool outputs are filtered from research extraction."""
    from deepresearch.nodes import think_tool

    messages = [
        HumanMessage(content="test topic"),
        AIMessage(content="searching..."),
        ToolMessage(content="Reflection recorded: evaluating", name=think_tool.name, tool_call_id="t1"),
        ToolMessage(content="Real search result from https://example.com/result", name="search_web", tool_call_id="t2"),
        AIMessage(content="Final synthesis with source https://example.com/result"),
    ]

    compressed, raw_notes, evidence_ledger = extract_research_from_messages({"messages": messages})
    assert compressed is not None
    assert "Reflection recorded" not in compressed
    assert raw_notes
    assert evidence_ledger
    assert not any("Reflection recorded" in note for note in raw_notes)


def test_extract_research_from_messages_empty():
    """Verify graceful handling of empty results."""
    compressed, raw_notes, evidence_ledger = extract_research_from_messages({"messages": []})
    assert compressed is None
    assert raw_notes == []
    assert evidence_ledger == []

    compressed, raw_notes, evidence_ledger = extract_research_from_messages({})
    assert compressed is None
    assert raw_notes == []
    assert evidence_ledger == []


def test_render_researcher_prompt_no_search_mode_omits_search_tool_contract():
    prompt = researcher_subgraph.render_researcher_prompt(search_enabled=False)

    assert "search" in prompt.lower() and "not available" in prompt.lower()
    assert "search_web" not in prompt or "not available" in prompt.lower()
