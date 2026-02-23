"""Native supervisor subgraph and integration node."""

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
    get_max_concurrent_research_units,
    get_max_react_tool_calls,
    get_max_researcher_iterations,
    get_supervisor_final_report_max_sections,
    get_supervisor_notes_max_bullets,
    get_supervisor_notes_word_budget,
)
from .nodes import think_tool
from .prompts import COMPRESSION_PROMPT, FINAL_REPORT_PROMPT, SUPERVISOR_PROMPT
from .researcher_subgraph import build_researcher_subgraph, extract_research_from_messages
from .runtime_utils import invoke_runnable_with_config, log_runtime_event
from .state import (
    ConductResearch,
    FALLBACK_CLARIFY_QUESTION,
    ResearchState,
    ResearchComplete,
    SupervisorState,
    extract_tool_calls,
    join_note_list,
    latest_ai_message,
    latest_human_text,
    normalize_note_list,
    state_text_or_none,
    stringify_tool_output,
    today_utc_date,
)

_logger = logging.getLogger(__name__)


def render_supervisor_prompt(current_date: str) -> str:
    supervisor_prompt = SUPERVISOR_PROMPT.format(
        current_date=current_date,
        max_concurrent_research_units=get_max_concurrent_research_units(),
        max_researcher_iterations=get_max_researcher_iterations(),
    )
    compression_prompt = COMPRESSION_PROMPT.format(
        notes_max_bullets=get_supervisor_notes_max_bullets(),
        notes_word_budget=get_supervisor_notes_word_budget(),
    )
    final_report_prompt = FINAL_REPORT_PROMPT.format(
        current_date=current_date,
        final_report_max_sections=get_supervisor_final_report_max_sections(),
    )
    return "\n\n".join((supervisor_prompt.strip(), compression_prompt.strip(), final_report_prompt.strip()))


async def supervisor(state: SupervisorState, config: RunnableConfig = None) -> dict[str, Any]:
    """Supervisor planning node with tool selection."""
    supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    research_brief = state_text_or_none(state.get("research_brief"))
    if not supervisor_messages and research_brief:
        supervisor_messages = [HumanMessage(content=research_brief)]

    context_blocks: list[str] = []
    notes = join_note_list(state.get("notes"))
    raw_notes = join_note_list(state.get("raw_notes"))
    if notes:
        context_blocks.append(f"Compressed notes so far:\n{notes}")
    if raw_notes:
        context_blocks.append(f"Raw notes so far:\n{raw_notes}")

    model_messages = [
        SystemMessage(content=render_supervisor_prompt(current_date=today_utc_date())),
        *supervisor_messages,
    ]
    if context_blocks:
        model_messages.append(HumanMessage(content="\n\n".join(context_blocks)))

    model = get_llm("orchestrator")
    if hasattr(model, "bind_tools"):
        model = model.bind_tools([ConductResearch, ResearchComplete, think_tool])

    response = await invoke_runnable_with_config(model, model_messages, config)
    if getattr(response, "type", "") != "ai":
        response = AIMessage(content=stringify_tool_output(response))

    return {"supervisor_messages": [response]}


async def invoke_single_tool(tool_obj: Any, args: dict[str, Any]) -> str:
    if hasattr(tool_obj, "ainvoke"):
        result = await tool_obj.ainvoke(args)
    else:
        result = tool_obj.invoke(args)
    return stringify_tool_output(result)


async def supervisor_tools(state: SupervisorState, config: RunnableConfig = None) -> dict[str, Any]:
    """Execute supervisor tools, including parallel researcher subgraph dispatch."""
    supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    latest_ai = latest_ai_message(supervisor_messages)
    if latest_ai is None:
        return {}

    tool_calls = extract_tool_calls(latest_ai)
    if not tool_calls:
        return {}

    think_calls = [call for call in tool_calls if call.get("name") == think_tool.name]
    research_calls = [call for call in tool_calls if call.get("name") == ConductResearch.__name__]
    complete_calls = [call for call in tool_calls if call.get("name") == ResearchComplete.__name__]
    research_iterations = int(state.get("research_iterations", 0) or 0)
    log_runtime_event(
        _logger,
        "supervisor_iteration",
        research_iterations=research_iterations,
        total_tool_calls=len(tool_calls),
        think_calls=len(think_calls),
        research_calls=len(research_calls),
        complete_calls=len(complete_calls),
    )

    tool_messages: list[ToolMessage] = []
    notes_additions: list[str] = []
    raw_notes_additions: list[str] = []
    completed = bool(complete_calls)

    async def run_think_call(index: int, call: dict[str, Any]) -> ToolMessage:
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        content = await invoke_single_tool(think_tool, args)
        return ToolMessage(
            content=content or "[No reflection recorded]",
            name=think_tool.name,
            tool_call_id=str(call.get("id") or f"supervisor_think_{index}"),
        )

    if think_calls:
        think_results = await asyncio.gather(
            *(run_think_call(index, call) for index, call in enumerate(think_calls)),
        )
        tool_messages.extend(think_results)

    remaining_iterations = max(0, get_max_researcher_iterations() - research_iterations)
    allowed_parallel = min(get_max_concurrent_research_units(), remaining_iterations)
    runnable_research_calls = research_calls[:allowed_parallel]
    skipped_research_calls = research_calls[allowed_parallel:]

    if runnable_research_calls:
        researcher_graph = build_researcher_subgraph()
        max_calls = get_max_react_tool_calls()
        researcher_recursion_limit = max_calls * 2 + 1

        async def run_research_call(index: int, call: dict[str, Any]) -> tuple[ToolMessage, str | None, list[str]]:
            call_id = str(call.get("id") or f"supervisor_research_{index}")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            topic = state_text_or_none(args.get("research_topic"))
            if not topic:
                return (
                    ToolMessage(
                        content="[ConductResearch missing research_topic]",
                        name=ConductResearch.__name__,
                        tool_call_id=call_id,
                    ),
                    None,
                    [],
                )

            payload = {"messages": [HumanMessage(content=topic)]}
            try:
                result = await researcher_graph.ainvoke(
                    payload,
                    config={"recursion_limit": researcher_recursion_limit},
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                failure_message = ToolMessage(
                    content=f"[Research unit failed: {exc}]",
                    name=ConductResearch.__name__,
                    tool_call_id=call_id,
                )
                return failure_message, None, []

            if not isinstance(result, dict):
                result = {}
            compressed, raw = extract_research_from_messages(result)
            content = compressed or join_note_list(raw) or "[Research unit returned no notes]"
            return (
                ToolMessage(content=content, name=ConductResearch.__name__, tool_call_id=call_id),
                compressed,
                raw,
            )

        research_results = await asyncio.gather(
            *(run_research_call(index, call) for index, call in enumerate(runnable_research_calls)),
        )
        for tool_message, compressed, raw in research_results:
            tool_messages.append(tool_message)
            if compressed:
                notes_additions.append(compressed)
            raw_notes_additions.extend(raw)
        log_runtime_event(
            _logger,
            "supervisor_research_dispatch",
            executed_calls=len(runnable_research_calls),
            skipped_calls=len(skipped_research_calls),
            remaining_iterations=remaining_iterations,
        )

    for index, call in enumerate(skipped_research_calls):
        tool_messages.append(
            ToolMessage(
                content=(
                    "[ConductResearch skipped: reached runtime cap "
                    f"(max_concurrent_research_units={get_max_concurrent_research_units()}, "
                    f"remaining_iterations={remaining_iterations})]"
                ),
                name=ConductResearch.__name__,
                tool_call_id=str(call.get("id") or f"supervisor_research_skipped_{index}"),
            )
        )

    for index, call in enumerate(complete_calls):
        tool_messages.append(
            ToolMessage(
                content="[ResearchComplete received]",
                name=ResearchComplete.__name__,
                tool_call_id=str(call.get("id") or f"supervisor_complete_{index}"),
            )
        )
    if completed:
        log_runtime_event(_logger, "supervisor_completion", research_iterations=research_iterations)

    return {
        "supervisor_messages": tool_messages,
        "notes": notes_additions,
        "raw_notes": raw_notes_additions,
        "research_iterations": (
            get_max_researcher_iterations()
            if completed
            else research_iterations + len(runnable_research_calls)
        ),
    }


def route_supervisor(state: SupervisorState) -> Literal["supervisor_tools", "__end__"]:
    """Route supervisor loop until no tools remain or runtime caps are reached."""
    if int(state.get("research_iterations", 0) or 0) >= get_max_researcher_iterations():
        return END

    supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    latest_ai = latest_ai_message(supervisor_messages)
    if latest_ai is None:
        return END

    return "supervisor_tools" if extract_tool_calls(latest_ai) else END


def route_supervisor_tools(state: SupervisorState) -> Literal["supervisor", "__end__"]:
    """Route back to supervisor unless iteration cap reached."""
    if int(state.get("research_iterations", 0) or 0) >= get_max_researcher_iterations():
        return END
    return "supervisor"


def build_supervisor_subgraph():
    """Build the native supervisor loop: supervisor -> supervisor_tools -> loop/end."""
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("supervisor_tools", supervisor_tools)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "supervisor_tools": "supervisor_tools",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "supervisor_tools",
        route_supervisor_tools,
        {
            "supervisor": "supervisor",
            END: END,
        },
    )
    return builder.compile()


async def research_supervisor(
    state: ResearchState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """Invoke native supervisor subgraph using the synthesized research brief."""
    messages = list(convert_to_messages(state.get("messages", [])))
    research_brief = state_text_or_none(state.get("research_brief")) or latest_human_text(messages)
    if not research_brief:
        return {
            "messages": [AIMessage(content=FALLBACK_CLARIFY_QUESTION)],
            "intake_decision": "clarify",
            "awaiting_clarification": True,
        }

    supervisor_graph = build_supervisor_subgraph()
    seed_supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    if not seed_supervisor_messages:
        seed_supervisor_messages = [HumanMessage(content=research_brief)]

    payload: dict[str, Any] = {
        "research_brief": research_brief,
        "supervisor_messages": seed_supervisor_messages,
        "notes": normalize_note_list(state.get("notes")),
        "raw_notes": normalize_note_list(state.get("raw_notes")),
        "research_iterations": 0,
    }

    try:
        result = await invoke_runnable_with_config(supervisor_graph, payload, config)
    except asyncio.CancelledError:
        raise
    except Exception:
        _logger.exception("research_supervisor failed; proceeding with existing note state")
        result = {}

    if not isinstance(result, dict):
        result = {}

    supervisor_result_messages = list(convert_to_messages(result.get("supervisor_messages", [])))
    supervisor_delta = supervisor_result_messages
    if seed_supervisor_messages and len(supervisor_result_messages) >= len(seed_supervisor_messages):
        supervisor_delta = supervisor_result_messages[len(seed_supervisor_messages) :]

    return {
        "supervisor_messages": supervisor_delta,
        "notes": normalize_note_list(result.get("notes")) or normalize_note_list(state.get("notes")),
        "raw_notes": normalize_note_list(result.get("raw_notes")) or normalize_note_list(state.get("raw_notes")),
        "intake_decision": "proceed",
        "awaiting_clarification": False,
    }
