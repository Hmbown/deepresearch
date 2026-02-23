"""Native researcher subgraph implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .config import (
    get_llm,
    get_max_react_tool_calls,
    get_researcher_complex_search_budget,
    get_researcher_simple_search_budget,
    get_search_tool,
)
from .nodes import _build_fetch_url_tool, _build_search_tool_with_processing, think_tool
from .prompts import RESEARCHER_PROMPT
from .runtime_utils import invoke_runnable_with_config, log_runtime_event
from .state import (
    ResearcherOutputState,
    ResearcherState,
    compress_note_text,
    extract_tool_calls,
    latest_ai_message,
    state_text_or_none,
    stringify_tool_output,
)

_logger = logging.getLogger(__name__)


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


async def researcher(state: ResearcherState, config: RunnableConfig = None) -> dict[str, Any]:
    """Researcher reasoning/tool-selection node."""
    tools = build_research_tools()
    model = get_llm("subagent")
    if hasattr(model, "bind_tools"):
        model = model.bind_tools(tools)

    messages = list(convert_to_messages(state.get("researcher_messages", [])))
    if not messages:
        topic = state_text_or_none(state.get("research_topic"))
        if topic:
            messages = [HumanMessage(content=topic)]
    log_runtime_event(
        _logger,
        "researcher_iteration",
        message_count=len(messages),
        tool_call_iterations=int(state.get("tool_call_iterations", 0) or 0),
    )
    payload = [SystemMessage(content=render_researcher_prompt()), *messages]
    response = await invoke_runnable_with_config(model, payload, config)
    if getattr(response, "type", "") != "ai":
        response = AIMessage(content=stringify_tool_output(response))
    return {"researcher_messages": [response]}


async def invoke_single_tool(tool_obj: Any, args: dict[str, Any]) -> str:
    if hasattr(tool_obj, "ainvoke"):
        result = await tool_obj.ainvoke(args)
    else:
        result = tool_obj.invoke(args)
    return stringify_tool_output(result)


async def researcher_tools(state: ResearcherState, config: RunnableConfig = None) -> dict[str, Any]:
    """Execute researcher tool calls in parallel with bounded call count."""
    del config
    messages = list(convert_to_messages(state.get("researcher_messages", [])))
    latest_ai = latest_ai_message(messages)
    if latest_ai is None:
        return {}

    tool_calls = extract_tool_calls(latest_ai)
    if not tool_calls:
        return {}

    current_calls = int(state.get("tool_call_iterations", 0) or 0)
    remaining = max(0, get_max_react_tool_calls() - current_calls)
    if remaining <= 0:
        return {}

    selected_calls = tool_calls[:remaining]
    log_runtime_event(
        _logger,
        "researcher_tool_batch",
        requested_calls=len(tool_calls),
        executed_calls=len(selected_calls),
        remaining_budget=remaining,
    )
    available_tools = {tool.name: tool for tool in build_research_tools()}

    async def execute_tool_call(index: int, call: dict[str, Any]) -> ToolMessage:
        tool_call_id = str(call.get("id") or f"researcher_tool_{index}")
        tool_name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}

        tool_obj = available_tools.get(tool_name)
        if tool_obj is None:
            log_runtime_event(_logger, "researcher_tool_call", tool=tool_name, status="unavailable")
            return ToolMessage(
                content=f"[Tool unavailable: {tool_name}]",
                name=tool_name,
                tool_call_id=tool_call_id,
            )

        try:
            output = await invoke_single_tool(tool_obj, args)
        except Exception as exc:  # pragma: no cover - defensive guard
            log_runtime_event(_logger, "researcher_tool_call", tool=tool_name, status="failed", error=str(exc))
            output = f"[Tool {tool_name} failed: {exc}]"
        else:
            log_runtime_event(_logger, "researcher_tool_call", tool=tool_name, status="ok")

        return ToolMessage(content=output or "[Tool returned empty output]", name=tool_name, tool_call_id=tool_call_id)

    tool_messages = await asyncio.gather(
        *(execute_tool_call(index, call) for index, call in enumerate(selected_calls)),
    )

    return {
        "researcher_messages": list(tool_messages),
        "tool_call_iterations": current_calls + len(selected_calls),
    }


def route_researcher(state: ResearcherState) -> Literal["researcher_tools", "compress_research"]:
    """Loop researcher tool usage until completion or runtime cap."""
    if int(state.get("tool_call_iterations", 0) or 0) >= get_max_react_tool_calls():
        return "compress_research"

    messages = list(convert_to_messages(state.get("researcher_messages", [])))
    latest_ai = latest_ai_message(messages)
    if latest_ai is None:
        return "compress_research"

    return "researcher_tools" if extract_tool_calls(latest_ai) else "compress_research"


async def compress_research(state: ResearcherState, config: RunnableConfig = None) -> dict[str, Any]:
    """Finalize one researcher unit into filtered compressed/raw notes."""
    del config
    messages = list(convert_to_messages(state.get("researcher_messages", [])))
    note_lines: list[str] = []
    for message in messages:
        if getattr(message, "type", "") not in {"ai", "tool"}:
            continue
        if getattr(message, "type", "") == "tool" and getattr(message, "name", "") == think_tool.name:
            continue
        line = stringify_tool_output(getattr(message, "content", "")).strip()
        if line:
            note_lines.append(line)

    raw_notes = "\n\n".join(note_lines).strip()
    compressed_research = compress_note_text(raw_notes)
    if compressed_research is None:
        compressed_research = state_text_or_none(raw_notes)

    return {
        "raw_notes": [state_text_or_none(raw_notes)] if state_text_or_none(raw_notes) else [],
        "compressed_research": state_text_or_none(compressed_research) or "",
    }


def build_researcher_subgraph():
    """Build the native researcher loop: researcher -> researcher_tools -> loop/compress."""
    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
    builder.add_node("researcher", researcher)
    builder.add_node("researcher_tools", researcher_tools)
    builder.add_node("compress_research", compress_research)

    builder.add_edge(START, "researcher")
    builder.add_conditional_edges(
        "researcher",
        route_researcher,
        {
            "researcher_tools": "researcher_tools",
            "compress_research": "compress_research",
        },
    )
    builder.add_edge("researcher_tools", "researcher")
    builder.add_edge("compress_research", END)
    return builder.compile()
