"""Native supervisor subgraph with Send-based research fan-out."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Literal
from urllib.parse import urlparse

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from .config import (
    get_llm,
    get_max_concurrent_research_units,
    get_max_researcher_iterations,
)
from .nodes import think_tool
from .prompts import SUPERVISOR_PROMPT
from .researcher_subgraph import build_researcher_subgraph, extract_research_from_messages
from .runtime_utils import invoke_runnable_with_config, log_runtime_event
from .state import (
    ConductResearch,
    FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH,
    EvidenceRecord,
    ResearchComplete,
    SupervisorState,
    filter_evidence_ledger,
    join_note_list,
    latest_ai_message,
    normalize_evidence_ledger,
    normalize_note_list,
    state_text_or_none,
    stringify_tool_output,
    today_utc_date,
)

_logger = logging.getLogger(__name__)
_RECURSION_LIMIT_PATTERN = re.compile(r"Recursion limit of \d+ reached")
_SUPERVISOR_PROGRESS_EVENT = "supervisor_progress"


def render_supervisor_prompt(current_date: str) -> str:
    return SUPERVISOR_PROMPT.format(
        current_date=current_date,
        max_concurrent_research_units=get_max_concurrent_research_units(),
        max_researcher_iterations=get_max_researcher_iterations(),
    )


def compute_research_dispatch_counts(
    *,
    requested_research_units: int,
    research_iterations: int,
) -> tuple[int, int]:
    """Return runnable-research count and remaining iteration budget."""
    bounded_requested = _coerce_non_negative_int(requested_research_units)
    bounded_iterations = _coerce_non_negative_int(research_iterations)
    remaining_iterations = max(0, get_max_researcher_iterations() - bounded_iterations)
    dispatch_count = min(bounded_requested, get_max_concurrent_research_units(), remaining_iterations)
    return dispatch_count, remaining_iterations


def _normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for idx, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            continue
        raw_args = raw_call.get("args")
        normalized.append(
            {
                "id": str(raw_call.get("id") or f"tool_call_{idx}"),
                "name": str(raw_call.get("name") or ""),
                "args": raw_args if isinstance(raw_args, dict) else {},
            }
        )
    return normalized


def _latest_supervisor_tool_calls(state: SupervisorState) -> list[dict[str, Any]]:
    supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    latest_ai = latest_ai_message(supervisor_messages)
    if latest_ai is None:
        return []
    return _normalize_tool_calls(getattr(latest_ai, "tool_calls", None))


def _has_any_tool_calls(messages: list[Any]) -> bool:
    for message in messages:
        if getattr(message, "type", "") != "ai":
            continue
        if _normalize_tool_calls(getattr(message, "tool_calls", None)):
            return True
    return False


def _extract_source_domains(evidence_ledger: list[EvidenceRecord]) -> list[str]:
    domains: set[str] = set()
    for record in evidence_ledger:
        for url in record.source_urls:
            parsed = urlparse(str(url))
            domain = (parsed.netloc or "").strip().lower()
            if domain:
                domains.add(domain)
    return sorted(domains)


def _dedupe_evidence_records(evidence_ledger: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """URL-level dedupe that prefers fetched provenance over model-cited."""
    by_url: dict[str, EvidenceRecord] = {}
    ordered_urls: list[str] = []

    for record in evidence_ledger:
        source_type = "fetched" if record.source_type == "fetched" else "model_cited"
        for raw_url in record.source_urls:
            url = str(raw_url).strip()
            if not url:
                continue
            existing = by_url.get(url)
            if existing is None:
                by_url[url] = EvidenceRecord(source_urls=[url], source_type=source_type)
                ordered_urls.append(url)
                continue
            if existing.source_type != "fetched" and source_type == "fetched":
                by_url[url] = EvidenceRecord(source_urls=[url], source_type="fetched")

    return [by_url[url] for url in ordered_urls]


def _sanitize_research_unit_failure(exc: Exception) -> str:
    """Return safe error text for tool messages without leaking internals."""
    if isinstance(exc, asyncio.TimeoutError):
        return "research unit timed out"
    match = _RECURSION_LIMIT_PATTERN.search(str(exc))
    if match:
        return match.group(0)
    return "research unit execution failed"


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _prepare_reset_payload() -> dict[str, Any]:
    return {
        "pending_research_calls": [],
        "pending_complete_calls": [],
        "pending_requested_research_units": 0,
        "pending_dispatched_research_units": 0,
        "pending_skipped_research_units": 0,
        "pending_remaining_iterations": 0,
    }


def _latest_ai_with_tool_calls(state: SupervisorState) -> AIMessage | None:
    supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    latest_ai = latest_ai_message(supervisor_messages)
    if latest_ai is None:
        return None
    if not _normalize_tool_calls(getattr(latest_ai, "tool_calls", None)):
        return None
    return latest_ai


def _partition_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    think_calls = [call for call in tool_calls if call.get("name") == think_tool.name]
    research_calls = [call for call in tool_calls if call.get("name") == ConductResearch.__name__]
    complete_calls = [call for call in tool_calls if call.get("name") == ResearchComplete.__name__]
    return think_calls, research_calls, complete_calls


async def _run_think_calls(think_calls: list[dict[str, Any]]) -> list[ToolMessage]:
    tool_messages: list[ToolMessage] = []
    for index, call in enumerate(think_calls):
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        content = await invoke_single_tool(think_tool, args)
        tool_messages.append(
            ToolMessage(
                content=content or "[No reflection recorded]",
                name=think_tool.name,
                tool_call_id=str(call.get("id") or f"supervisor_think_{index}"),
            )
        )
    return tool_messages


def _prepare_research_calls(
    runnable_research_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared_research_calls: list[dict[str, Any]] = []
    for index, call in enumerate(runnable_research_calls):
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        prepared_research_calls.append(
            {
                "id": str(call.get("id") or f"supervisor_research_{index}"),
                "args": args,
                "topic": state_text_or_none(args.get("research_topic")) or "",
            }
        )
    return prepared_research_calls


def _prepare_complete_call_payloads(complete_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete_call_payloads: list[dict[str, Any]] = []
    for index, call in enumerate(complete_calls):
        complete_call_payloads.append(
            {
                "id": str(call.get("id") or f"supervisor_complete_{index}"),
            }
        )
    return complete_call_payloads


def _build_skipped_research_messages(
    skipped_research_calls: list[dict[str, Any]],
    remaining_iterations: int,
) -> tuple[list[ToolMessage], list[dict[str, Any]]]:
    tool_messages: list[ToolMessage] = []
    skipped_summaries: list[dict[str, Any]] = []
    max_concurrent_research_units = get_max_concurrent_research_units()
    for index, call in enumerate(skipped_research_calls):
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        topic = state_text_or_none(args.get("research_topic")) or ""
        tool_call_id = str(call.get("id") or f"supervisor_research_skipped_{index}")
        tool_messages.append(
            ToolMessage(
                content=(
                    "[ConductResearch skipped: reached runtime cap "
                    f"(max_concurrent_research_units={max_concurrent_research_units}, "
                    f"remaining_iterations={remaining_iterations})]"
                ),
                name=ConductResearch.__name__,
                tool_call_id=tool_call_id,
            )
        )
        skipped_summaries.append(
            {
                "call_id": tool_call_id,
                "topic": topic,
                "status": "skipped",
                "failure_reason": "reached runtime cap",
                "evidence_record_count": 0,
                "source_domain_count": 0,
                "duration_seconds": 0.0,
            }
        )
    return tool_messages, skipped_summaries


def _build_prepare_update(
    *,
    think_messages: list[ToolMessage],
    skipped_messages: list[ToolMessage],
    skipped_summaries: list[dict[str, Any]],
    research_calls: list[dict[str, Any]],
    prepared_research_calls: list[dict[str, Any]],
    complete_call_payloads: list[dict[str, Any]],
    remaining_iterations: int,
) -> dict[str, Any]:
    return {
        "supervisor_messages": [*think_messages, *skipped_messages],
        "pending_research_calls": prepared_research_calls,
        "pending_complete_calls": complete_call_payloads,
        "pending_requested_research_units": len(research_calls),
        "pending_dispatched_research_units": len(prepared_research_calls),
        "pending_skipped_research_units": len(research_calls) - len(prepared_research_calls),
        "pending_remaining_iterations": remaining_iterations,
        "research_unit_summaries": skipped_summaries,
    }


async def supervisor(state: SupervisorState, config: RunnableConfig = None) -> dict[str, Any]:
    """Supervisor planning node with tool selection."""
    supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    research_brief = state_text_or_none(state.get("research_brief"))
    if not supervisor_messages and research_brief:
        supervisor_messages = [HumanMessage(content=research_brief)]

    model_messages = [
        SystemMessage(content=render_supervisor_prompt(current_date=today_utc_date())),
        *supervisor_messages,
    ]

    model = get_llm("orchestrator")
    if hasattr(model, "bind_tools"):
        model = model.bind_tools([ConductResearch, ResearchComplete, think_tool])

    try:
        response = await invoke_runnable_with_config(model, model_messages, config)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _logger.exception("supervisor planning failed; activating deterministic fallback")
        return {
            "supervisor_exception": str(exc),
            "research_iterations": get_max_researcher_iterations(),
        }

    if getattr(response, "type", "") != "ai":
        response = AIMessage(content=stringify_tool_output(response))

    return {
        "supervisor_messages": [response],
        "supervisor_exception": None,
    }


async def invoke_single_tool(tool_obj: Any, args: dict[str, Any]) -> str:
    if hasattr(tool_obj, "ainvoke"):
        result = await tool_obj.ainvoke(args)
    else:
        result = tool_obj.invoke(args)
    return stringify_tool_output(result)


async def supervisor_prepare(state: SupervisorState, config: RunnableConfig = None) -> dict[str, Any]:
    """Prepare one supervisor tool cycle and compute research dispatch inputs."""
    del config
    latest_ai = _latest_ai_with_tool_calls(state)
    if latest_ai is None:
        return _prepare_reset_payload()

    tool_calls = _normalize_tool_calls(getattr(latest_ai, "tool_calls", None))
    think_calls, research_calls, complete_calls = _partition_tool_calls(tool_calls)
    research_iterations = _coerce_non_negative_int(state.get("research_iterations", 0))
    log_runtime_event(
        _logger,
        "supervisor_iteration",
        research_iterations=research_iterations,
        total_tool_calls=len(tool_calls),
        think_calls=len(think_calls),
        research_calls=len(research_calls),
        complete_calls=len(complete_calls),
    )

    think_messages = await _run_think_calls(think_calls)

    dispatch_count, remaining_iterations = compute_research_dispatch_counts(
        requested_research_units=len(research_calls),
        research_iterations=research_iterations,
    )
    runnable_research_calls = research_calls[:dispatch_count]
    skipped_research_calls = research_calls[dispatch_count:]

    prepared_research_calls = _prepare_research_calls(runnable_research_calls)
    complete_call_payloads = _prepare_complete_call_payloads(complete_calls)
    skipped_messages, skipped_summaries = _build_skipped_research_messages(
        skipped_research_calls,
        remaining_iterations,
    )

    if prepared_research_calls:
        log_runtime_event(
            _logger,
            "supervisor_research_dispatch",
            executed_calls=len(prepared_research_calls),
            skipped_calls=len(skipped_research_calls),
            remaining_iterations=remaining_iterations,
        )

    return _build_prepare_update(
        think_messages=think_messages,
        skipped_messages=skipped_messages,
        skipped_summaries=skipped_summaries,
        research_calls=research_calls,
        prepared_research_calls=prepared_research_calls,
        complete_call_payloads=complete_call_payloads,
        remaining_iterations=remaining_iterations,
    )


def route_supervisor_prepare(
    state: SupervisorState,
) -> list[Send] | Literal["supervisor_finalize"]:
    """Fan out ConductResearch tool calls with native Send dispatch."""
    pending_calls = state.get("pending_research_calls")
    if not isinstance(pending_calls, list) or not pending_calls:
        return "supervisor_finalize"

    sends: list[Send] = []
    for call in pending_calls:
        if not isinstance(call, dict):
            continue
        sends.append(Send("run_research_unit", {"research_call": call}))
    if sends:
        return sends
    return "supervisor_finalize"


def research_barrier(state: SupervisorState) -> Command[Literal["supervisor_finalize"]]:
    """Wait until all dispatched research units in the current wave have returned."""
    dispatched_research_units = _coerce_non_negative_int(state.get("pending_dispatched_research_units", 0))
    if dispatched_research_units <= 0:
        return Command(goto="supervisor_finalize")

    summaries_raw = state.get("research_unit_summaries")
    summaries = summaries_raw if isinstance(summaries_raw, list) else []
    consumed = _coerce_non_negative_int(state.get("research_unit_summaries_consumed", 0))
    completed_research_units = max(0, len(summaries) - consumed)

    if completed_research_units < dispatched_research_units:
        log_runtime_event(
            _logger,
            "supervisor_research_barrier_waiting",
            completed_research_units=completed_research_units,
            dispatched_research_units=dispatched_research_units,
        )
        return Command(update={})

    return Command(goto="supervisor_finalize")


async def run_research_unit(state: dict[str, Any]) -> dict[str, Any]:
    """Execute one ConductResearch call from Send fan-out."""
    def _missing_topic_result(call_id: str, duration_seconds: float) -> dict[str, Any]:
        return {
            "supervisor_messages": [
                ToolMessage(
                    content="[ConductResearch missing research_topic]",
                    name=ConductResearch.__name__,
                    tool_call_id=call_id,
                )
            ],
            "research_unit_summaries": [
                {
                    "call_id": call_id,
                    "topic": "",
                    "status": "missing_topic",
                    "failure_reason": "missing research_topic",
                    "evidence_record_count": 0,
                    "source_domain_count": 0,
                    "duration_seconds": duration_seconds,
                }
            ],
        }

    call = state.get("research_call") if isinstance(state, dict) else None
    if not isinstance(call, dict):
        return _missing_topic_result("supervisor_research_missing_call", 0.0)

    call_id = str(call.get("id") or "supervisor_research")
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    topic = state_text_or_none(call.get("topic")) or state_text_or_none(args.get("research_topic"))
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    if not topic:
        return _missing_topic_result(call_id, round(loop.time() - started_at, 3))

    researcher_graph = build_researcher_subgraph()
    payload = {"messages": [HumanMessage(content=topic)]}
    try:
        result = await researcher_graph.ainvoke(payload)
    except Exception as exc:  # pragma: no cover - defensive guard
        failure_reason = _sanitize_research_unit_failure(exc)
        return {
            "supervisor_messages": [
                ToolMessage(
                    content=f"[Research unit failed: {failure_reason}]",
                    name=ConductResearch.__name__,
                    tool_call_id=call_id,
                )
            ],
            "research_unit_summaries": [
                {
                    "call_id": call_id,
                    "topic": topic,
                    "status": "failed",
                    "failure_reason": failure_reason,
                    "evidence_record_count": 0,
                    "source_domain_count": 0,
                    "duration_seconds": round(asyncio.get_running_loop().time() - started_at, 3),
                }
            ],
        }

    if not isinstance(result, dict):
        result = {}

    compressed, raw, evidence = extract_research_from_messages(result)
    fetched_evidence = filter_evidence_ledger(evidence, source_type="fetched")
    model_cited_evidence = filter_evidence_ledger(evidence, source_type="model_cited")
    content = compressed or join_note_list(raw) or "[Research unit returned no notes]"
    source_domains = _extract_source_domains(fetched_evidence)
    status = "empty" if not compressed and not raw and not evidence else "completed"
    updates: dict[str, Any] = {
        "supervisor_messages": [
            ToolMessage(
                content=content,
                name=ConductResearch.__name__,
                tool_call_id=call_id,
            )
        ],
        "raw_notes": raw,
        "evidence_ledger": evidence,
        "research_unit_summaries": [
            {
                "call_id": call_id,
                "topic": topic,
                "status": status,
                "evidence_record_count": len(fetched_evidence),
                "source_domain_count": len(source_domains),
                "model_cited_record_count": len(model_cited_evidence),
                "duration_seconds": round(asyncio.get_running_loop().time() - started_at, 3),
            }
        ],
    }
    if compressed:
        updates["notes"] = [compressed]
    return updates


async def supervisor_finalize(state: SupervisorState, config: RunnableConfig = None) -> dict[str, Any]:
    """Finalize one supervisor cycle and emit progress telemetry."""
    research_iterations = _coerce_non_negative_int(state.get("research_iterations", 0))
    requested_research_units = _coerce_non_negative_int(state.get("pending_requested_research_units", 0))
    dispatched_research_units = _coerce_non_negative_int(state.get("pending_dispatched_research_units", 0))
    skipped_research_units = _coerce_non_negative_int(state.get("pending_skipped_research_units", 0))
    remaining_iterations = _coerce_non_negative_int(state.get("pending_remaining_iterations", 0))
    planned_research_calls_raw = state.get("pending_research_calls")
    planned_research_calls = (
        [call for call in planned_research_calls_raw if isinstance(call, dict)]
        if isinstance(planned_research_calls_raw, list)
        else []
    )

    complete_calls_raw = state.get("pending_complete_calls")
    complete_calls = (
        [call for call in complete_calls_raw if isinstance(call, dict)]
        if isinstance(complete_calls_raw, list)
        else []
    )

    existing_evidence = _dedupe_evidence_records(normalize_evidence_ledger(state.get("evidence_ledger")))
    fetched_evidence = filter_evidence_ledger(existing_evidence, source_type="fetched")
    model_cited_evidence = filter_evidence_ledger(existing_evidence, source_type="model_cited")
    source_domains = _extract_source_domains(fetched_evidence)
    model_cited_domains = _extract_source_domains(model_cited_evidence)

    completed = bool(complete_calls)
    tool_messages: list[ToolMessage] = []
    if complete_calls:
        log_runtime_event(
            _logger,
            "supervisor_research_complete",
            evidence_record_count=len(fetched_evidence),
            source_domain_count=len(source_domains),
            model_cited_record_count=len(model_cited_evidence),
            model_cited_domain_count=len(model_cited_domains),
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

    max_researcher_iterations = get_max_researcher_iterations()
    next_research_iterations = research_iterations + dispatched_research_units
    if completed:
        next_research_iterations = max_researcher_iterations

    all_summaries = state.get("research_unit_summaries")
    summary_items = all_summaries if isinstance(all_summaries, list) else []
    consumed = _coerce_non_negative_int(state.get("research_unit_summaries_consumed", 0))
    summary_slice = summary_items[consumed:] if consumed < len(summary_items) else []
    research_units = [summary for summary in summary_slice if isinstance(summary, dict)]

    progress_payload = {
        "supervisor_iteration": research_iterations + 1,
        "requested_research_units": requested_research_units,
        "dispatched_research_units": dispatched_research_units,
        "skipped_research_units": skipped_research_units,
        "remaining_iterations": remaining_iterations,
        "max_concurrent_research_units": get_max_concurrent_research_units(),
        "max_researcher_iterations": max_researcher_iterations,
        "evidence_record_count": len(fetched_evidence),
        "source_domain_count": len(source_domains),
        "source_domains": sorted(source_domains),
        "model_cited_record_count": len(model_cited_evidence),
        "model_cited_domain_count": len(model_cited_domains),
        "model_cited_domains": sorted(model_cited_domains),
        "planned_research_units": [
            {
                "call_id": str(call.get("id") or ""),
                "topic": state_text_or_none(call.get("topic")) or "",
            }
            for call in planned_research_calls
        ],
        "research_units": research_units,
    }
    try:
        await adispatch_custom_event(_SUPERVISOR_PROGRESS_EVENT, progress_payload, config=config)
    except RuntimeError:
        # Direct unit invocations (outside a LangGraph run context) do not have a parent run id.
        pass

    return {
        "supervisor_messages": tool_messages,
        "research_iterations": next_research_iterations,
        "research_unit_summaries_consumed": consumed + len(summary_slice),
        **_prepare_reset_payload(),
    }


def supervisor_route(state: SupervisorState) -> Literal["supervisor_prepare", "supervisor_terminal"]:
    """Route supervisor loop until no tools remain or runtime caps are reached."""
    if state_text_or_none(state.get("supervisor_exception")):
        return "supervisor_terminal"

    if _coerce_non_negative_int(state.get("research_iterations", 0)) >= get_max_researcher_iterations():
        return "supervisor_terminal"

    return "supervisor_prepare" if _latest_supervisor_tool_calls(state) else "supervisor_terminal"


def supervisor_finalize_route(state: SupervisorState) -> Literal["supervisor", "supervisor_terminal"]:
    """Route back to supervisor unless iteration cap is reached."""
    if state_text_or_none(state.get("supervisor_exception")):
        return "supervisor_terminal"

    if _coerce_non_negative_int(state.get("research_iterations", 0)) >= get_max_researcher_iterations():
        return "supervisor_terminal"

    return "supervisor"


async def supervisor_terminal(state: SupervisorState) -> dict[str, Any]:
    """Finalize the supervisor run and emit deterministic fallback when needed."""
    supervisor_messages = list(convert_to_messages(state.get("supervisor_messages", [])))
    supervisor_exception = state_text_or_none(state.get("supervisor_exception"))
    research_iterations = _coerce_non_negative_int(state.get("research_iterations", 0))
    max_iterations = get_max_researcher_iterations()

    has_tool_calls = _has_any_tool_calls(supervisor_messages)
    has_any_notes = bool(normalize_note_list(state.get("notes"))) or bool(normalize_note_list(state.get("raw_notes")))
    has_any_evidence = bool(normalize_evidence_ledger(state.get("evidence_ledger")))

    # Log termination reason for observability
    if research_iterations >= max_iterations and has_any_evidence:
        log_runtime_event(
            _logger,
            "supervisor_iteration_cap_reached",
            research_iterations=research_iterations,
            max_researcher_iterations=max_iterations,
            had_notes=has_any_notes,
            had_evidence=has_any_evidence,
        )

    if supervisor_exception or (not has_any_notes and not has_any_evidence):
        fallback_reason = "exception" if supervisor_exception else "no_notes"
        log_runtime_event(
            _logger,
            "supervisor_no_useful_research_fallback",
            reason=fallback_reason,
            had_tool_calls=has_tool_calls,
            supervisor_message_count=len(supervisor_messages),
            error=supervisor_exception,
        )
        fallback_text = FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH
        return {
            "messages": [AIMessage(content=fallback_text)],
            "notes": [],
            "raw_notes": [],
            "evidence_ledger": [],
            "final_report": fallback_text,
            "intake_decision": "proceed",
            "awaiting_clarification": False,
        }

    return {
        "intake_decision": "proceed",
        "awaiting_clarification": False,
    }


def build_supervisor_subgraph():
    """Build the native supervisor loop with Send-based research fan-out."""
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("supervisor_prepare", supervisor_prepare)
    builder.add_node("run_research_unit", run_research_unit)
    builder.add_node("research_barrier", research_barrier)
    builder.add_node("supervisor_finalize", supervisor_finalize)
    builder.add_node("supervisor_terminal", supervisor_terminal)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "supervisor_prepare": "supervisor_prepare",
            "supervisor_terminal": "supervisor_terminal",
        },
    )
    builder.add_conditional_edges(
        "supervisor_prepare",
        route_supervisor_prepare,
        {
            "supervisor_finalize": "supervisor_finalize",
        },
    )
    builder.add_edge("run_research_unit", "research_barrier")
    builder.add_conditional_edges(
        "supervisor_finalize",
        supervisor_finalize_route,
        {
            "supervisor": "supervisor",
            "supervisor_terminal": "supervisor_terminal",
        },
    )
    builder.add_edge("supervisor_terminal", END)
    return builder.compile()
