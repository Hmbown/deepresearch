import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from deepresearch import researcher_subgraph


class _FakeResearcherLoopModel:
    def __init__(self):
        self.calls = 0
        self.observed_search_output = False
        self.observed_fetch_output = False

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, config=None):
        del config
        self.calls += 1
        tool_texts = [
            str(getattr(message, "content", ""))
            for message in messages
            if getattr(message, "type", "") == "tool"
        ]

        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search-1",
                        "name": "search_web",
                        "args": {"query": "deterministic integration test query"},
                    }
                ],
            )
        if self.calls == 2:
            self.observed_search_output = any("https://example.com/source-a" in text for text in tool_texts)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "fetch-1",
                        "name": "fetch_url",
                        "args": {"url": "https://example.com/source-a"},
                    }
                ],
            )
        if self.calls == 3:
            self.observed_fetch_output = any("Fetched detail [2]" in text for text in tool_texts)
            return AIMessage(
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
            )
        return AIMessage(content="done")


@tool("search_web")
async def _fake_search_web(query: str) -> str:
    """Return deterministic search output with a source URL and citation marker."""
    del query
    return (
        "[Source 1] Example Source A\n"
        "URL: https://example.com/source-a\n"
        "Summary: Baseline measurement [1]."
    )


@tool("fetch_url")
async def _fake_fetch_url(url: str) -> str:
    """Return deterministic fetched content with a source citation marker."""
    del url
    return "Fetched detail [2] from https://example.com/source-b"


def test_researcher_subgraph_real_loop_runs_multiple_iterations_and_preserves_sources(monkeypatch):
    model = _FakeResearcherLoopModel()
    monkeypatch.setattr(researcher_subgraph, "get_llm", lambda role: model)
    monkeypatch.setattr(
        researcher_subgraph,
        "build_research_tools",
        lambda writer=None: [_fake_search_web, _fake_fetch_url, researcher_subgraph.think_tool],
    )

    graph = researcher_subgraph.build_researcher_subgraph()
    result = asyncio.run(
        graph.ainvoke(
            {
                "researcher_messages": [HumanMessage(content="Research integration topic")],
                "tool_call_iterations": 0,
                "research_topic": "Research integration topic",
                "compressed_research": "",
                "raw_notes": [],
            }
        )
    )

    assert model.calls >= 3
    assert model.observed_search_output is True
    assert model.observed_fetch_output is True
    assert any("https://example.com/source-a" in note for note in result["raw_notes"])
    assert any("[1]" in note or "[2]" in note for note in result["raw_notes"])
    assert "[1]" in result["compressed_research"] or "https://example.com/source-a" in result["compressed_research"]
