"""Researcher subgraph backed by a deep agent with built-in middleware."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from .config import (
    get_max_react_tool_calls,
    get_model_string,
    get_researcher_complex_search_budget,
    get_researcher_simple_search_budget,
    get_search_tool,
)
from .nodes import _build_fetch_url_tool, _build_search_tool_with_processing, think_tool
from .prompts import RESEARCHER_PROMPT
from .state import compress_note_text, stringify_tool_output


def build_research_tools(writer: Any | None = None) -> list[Any]:
    base_search_tool = get_search_tool()
    fetch_url_tool = _build_fetch_url_tool(writer)
    tools = [think_tool, fetch_url_tool]
    if base_search_tool is not None:
        tools.insert(0, _build_search_tool_with_processing(base_search_tool, writer))
    return tools


def render_researcher_prompt() -> str:
    return RESEARCHER_PROMPT.format(
        researcher_simple_search_budget=get_researcher_simple_search_budget(),
        researcher_complex_search_budget=get_researcher_complex_search_budget(),
        max_react_tool_calls=get_max_react_tool_calls(),
    )


def build_researcher_subgraph():
    """Build a deep-agent researcher with built-in middleware."""
    model_str = get_model_string("subagent")
    tools = build_research_tools()
    system_prompt = render_researcher_prompt()

    return create_deep_agent(
        model=model_str,
        tools=tools,
        system_prompt=system_prompt,
        name="deep-researcher",
    )


def extract_research_from_messages(result: dict) -> tuple[str | None, list[str]]:
    """Post-process deep agent MessagesState into compressed_research + raw_notes."""
    messages = result.get("messages", [])
    note_lines: list[str] = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type not in {"ai", "tool"}:
            continue
        if msg_type == "tool" and getattr(msg, "name", "") == think_tool.name:
            continue
        line = stringify_tool_output(getattr(msg, "content", "")).strip()
        if line:
            note_lines.append(line)

    raw_text = "\n\n".join(note_lines).strip()
    compressed = compress_note_text(raw_text)
    raw_notes = [raw_text] if raw_text else []
    return compressed or raw_text or None, raw_notes
