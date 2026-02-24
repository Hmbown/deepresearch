from __future__ import annotations

from types import SimpleNamespace

from deepresearch import cli


def test_parse_search_summary_extracts_counts_and_domains():
    raw = """
[Source 1] Gemini deep research
URL: https://blog.google/ai/research
[Source 2] OpenAI
URL: https://openai.com/research
"""

    count, domains = cli._parse_search_summary(raw)
    assert count == 2
    assert domains == ["blog.google", "openai.com"]


def test_parse_search_summary_returns_zero_when_no_results():
    assert cli._parse_search_summary("No relevant search results found.") == (0, [])


def test_format_domain_list_truncates():
    domains = ["a.com", "b.org", "c.net", "d.io", "e.ai"]
    assert cli._format_domain_list(domains, max_items=3) == "a.com, b.org, c.net, ..."


def test_extract_tool_message_summary_for_quality_gate():
    assert cli._extract_tool_message_summary(
        SimpleNamespace(content="[ResearchComplete rejected: evidence quality gate failed (reason=insufficient_source_domains)]"),
    ) == ("research_complete_rejected", "insufficient_source_domains")

    assert cli._extract_tool_message_summary(
        SimpleNamespace(content="[ResearchComplete received]")
    ) == ("research_complete_received", "")


def test_extract_research_topics_from_supervisor_input():
    message = SimpleNamespace(
        type="ai",
        tool_calls=[
            {
                "name": "ConductResearch",
                "id": "a",
                "args": {"research_topic": "LangGraph agent architecture"},
            },
            {"name": "think_tool", "id": "b", "args": {"reflection": "skip"}},
            {
                "name": "ConductResearch",
                "id": "c",
                "args": {"research_topic": "LLM evaluation"},
            },
        ],
    )

    assert cli._extract_research_topics_from_supervisor_input(
        {"supervisor_messages": [message]}
    ) == ["LangGraph agent architecture", "LLM evaluation"]
