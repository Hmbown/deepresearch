"""Intake scoping and brief generation for supervisor handoff."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from .config import get_llm, get_max_concurrent_research_units, get_max_structured_output_retries
from .prompts import CLARIFY_PROMPT, RESEARCH_BRIEF_PROMPT, RESEARCH_PLAN_PROMPT
from .runtime_utils import invoke_structured_with_retries
from .state import (
    FALLBACK_CLARIFY_QUESTION,
    FALLBACK_FOLLOWUP_CLARIFICATION,
    FALLBACK_PLAN_CONFIRMATION_FOOTER,
    FALLBACK_SCOPE_BOUNDARY_CLARIFICATION,
    FALLBACK_VERIFICATION,
    ClarifyWithUser,
    ResearchBrief,
    ResearchPlan,
    ResearchState,
    has_scope_boundary,
    is_broad_scope_request,
    latest_human_text,
    should_recheck_intent_on_follow_up,
    today_utc_date,
)

_PLAN_ACK_NEGATED_POSITIVE_RE = re.compile(r"\b(?:don't|do not|not)\s+(?:start|proceed|continue|launch)\b")
_PLAN_ACK_STRONG_POSITIVE_RE = re.compile(r"\b(?:start|proceed|continue|launch)\b")
_PLAN_ACK_GO_AHEAD_RE = re.compile(r"\bgo\s+ahead\b")
_PLAN_ACK_LENIENT_POSITIVE_RE = re.compile(r"\b(?:yes|yeah|yep|sure|ok|okay)\b")
_PLAN_ACK_HOLD_OR_REVISE_RE = re.compile(r"\b(?:wait|pause|hold|stop|revise|change|adjust|update|clarify)\b|not yet")
_PLAN_ACK_POSITIVE_PHRASES = (
    "sounds good",
    "looks good",
    "works for me",
    "all good",
    "good to go",
)


def _is_follow_up_topic_shift(state: ResearchState, messages: list[Any]) -> bool:
    return state.get("intake_decision") == "proceed" and should_recheck_intent_on_follow_up(messages)


def _build_topic_shift_reset_update() -> dict[str, Any]:
    return {
        "research_brief": None,
        "supervisor_messages": [],
        "notes": [],
        "raw_notes": [],
        "evidence_ledger": [],
        "final_report": "",
    }


def _should_run_clarification_pass(state: ResearchState, messages: list[Any]) -> bool:
    if bool(state.get("awaiting_clarification")):
        return True

    intake_decision = state.get("intake_decision")
    if intake_decision is None:
        return not bool(state.get("research_brief"))

    if intake_decision == "proceed":
        return _is_follow_up_topic_shift(state, messages)

    return True


def _needs_scope_boundary_clarification(messages: list[Any]) -> bool:
    return is_broad_scope_request(messages) and not has_scope_boundary(messages)


def _format_plan_message(plan: ResearchPlan) -> str:
    """Format a structured research plan for user confirmation."""
    lines = ["Before I start research, here is the plan:\n"]
    lines.append(f"**Scope:** {plan.scope}\n")
    lines.append("**Research tracks:**")
    for i, track in enumerate(plan.research_tracks, 1):
        lines.append(f"  {i}. {track}")
    lines.append(f"\n**Evidence strategy:** {plan.evidence_strategy}")
    lines.append(f"**Output:** {plan.output_format}\n")
    lines.append(FALLBACK_PLAN_CONFIRMATION_FOOTER)
    return "\n".join(lines)


async def _generate_research_plan(
    research_brief: str,
    config: RunnableConfig | None = None,
) -> ResearchPlan | None:
    """Generate a structured research plan from the brief."""
    structured_model = get_llm("orchestrator").with_structured_output(ResearchPlan)
    prompt_content = RESEARCH_PLAN_PROMPT.format(
        research_brief=research_brief,
        date=today_utc_date(),
        max_research_tracks=get_max_concurrent_research_units(),
    )
    return await invoke_structured_with_retries(
        structured_model,
        prompt_content,
        config,
        schema_name="ResearchPlan",
        max_retries=get_max_structured_output_retries(),
    )


async def _generate_research_brief(messages: list[Any], config: RunnableConfig | None = None) -> str:
    """Generate a focused research brief from full chat history."""
    structured_model = get_llm("orchestrator").with_structured_output(ResearchBrief)
    prompt_content = RESEARCH_BRIEF_PROMPT.format(
        messages=get_buffer_string(messages),
        date=today_utc_date(),
    )
    response = await invoke_structured_with_retries(
        structured_model,
        prompt_content,
        config,
        schema_name="ResearchBrief",
        max_retries=get_max_structured_output_retries(),
    )

    research_brief = str(getattr(response, "research_brief", "")).strip()
    if not research_brief:
        research_brief = latest_human_text(messages)
    return research_brief


def _build_research_handoff_update(research_brief: str) -> dict[str, Any]:
    supervisor_messages = [HumanMessage(content=research_brief)] if research_brief else []
    return {
        "research_brief": research_brief,
        "supervisor_messages": supervisor_messages,
        "notes": [],
        "raw_notes": [],
        "evidence_ledger": [],
        "final_report": "",
        "intake_decision": "proceed",
        "awaiting_clarification": False,
    }


def _state_messages(state: ResearchState) -> list[Any]:
    return list(convert_to_messages(state.get("messages", [])))


def _is_plan_review_turn(state: ResearchState) -> bool:
    return bool(
        state.get("intake_decision") == "clarify"
        and state.get("research_brief")
        and state.get("awaiting_clarification")
    )


def _is_plan_acknowledgement_message(messages: list[Any]) -> bool:
    latest_text = re.sub(r"\s+", " ", latest_human_text(messages).lower()).strip()
    if not latest_text:
        return False
    if _PLAN_ACK_NEGATED_POSITIVE_RE.search(latest_text):
        return False

    has_revision_or_hold_marker = _PLAN_ACK_HOLD_OR_REVISE_RE.search(latest_text) is not None
    no_changes_phrase = "no change" in latest_text or "no changes" in latest_text
    if has_revision_or_hold_marker and not no_changes_phrase:
        return False

    if _PLAN_ACK_STRONG_POSITIVE_RE.search(latest_text):
        return True
    if _PLAN_ACK_GO_AHEAD_RE.search(latest_text):
        return True
    if _PLAN_ACK_LENIENT_POSITIVE_RE.search(latest_text):
        return True
    return any(phrase in latest_text for phrase in _PLAN_ACK_POSITIVE_PHRASES)


def _is_plan_acknowledgement_turn(state: ResearchState, messages: list[Any]) -> bool:
    return _is_plan_review_turn(state) and _is_plan_acknowledgement_message(messages)


async def _handle_plan_acknowledgement(
    messages: list[Any],
    config: RunnableConfig | None,
) -> Command[Literal["research_supervisor", "__end__"]]:
    research_brief = await _generate_research_brief(messages, config)
    return Command(
        goto="research_supervisor",
        update=_build_research_handoff_update(research_brief),
    )


async def _handle_direct_proceed(
    messages: list[Any],
    config: RunnableConfig | None,
) -> Command[Literal["research_supervisor", "__end__"]]:
    research_brief = await _generate_research_brief(messages, config)
    return Command(
        goto="research_supervisor",
        update=_build_research_handoff_update(research_brief),
    )


def _missing_user_text_command() -> Command[Literal["research_supervisor", "__end__"]]:
    return Command(
        goto=END,
        update={
            "messages": [AIMessage(content=FALLBACK_CLARIFY_QUESTION)],
            "intake_decision": "clarify",
            "awaiting_clarification": True,
        },
    )


def _scope_boundary_clarification_command(
    *,
    follow_up_topic_shift: bool,
) -> Command[Literal["research_supervisor", "__end__"]]:
    update = {
        "messages": [AIMessage(content=FALLBACK_SCOPE_BOUNDARY_CLARIFICATION)],
        "intake_decision": "clarify",
        "awaiting_clarification": True,
    }
    if follow_up_topic_shift:
        update.update(_build_topic_shift_reset_update())
    return Command(goto=END, update=update)


async def _invoke_clarification_check(
    messages: list[Any],
    config: RunnableConfig | None,
) -> Any:
    structured_model = get_llm("orchestrator").with_structured_output(ClarifyWithUser)
    prompt_content = CLARIFY_PROMPT.format(
        messages=get_buffer_string(messages),
        date=today_utc_date(),
    )
    response = await invoke_structured_with_retries(
        structured_model,
        prompt_content,
        config,
        schema_name="ClarifyWithUser",
        max_retries=get_max_structured_output_retries(),
    )
    if response is None:
        return SimpleNamespace(
            need_clarification=True,
            question=FALLBACK_FOLLOWUP_CLARIFICATION,
            verification="",
        )
    return response


def _clarification_command(
    question: str,
    *,
    follow_up_topic_shift: bool,
) -> Command[Literal["research_supervisor", "__end__"]]:
    update = {
        "messages": [AIMessage(content=question)],
        "intake_decision": "clarify",
        "awaiting_clarification": True,
    }
    if follow_up_topic_shift:
        update.update(_build_topic_shift_reset_update())
    return Command(goto=END, update=update)


def _plan_message(plan: ResearchPlan | None, research_brief: str) -> str:
    if plan is not None:
        return _format_plan_message(plan)
    return (
        f"Before I start research, here is the scope:\n\n"
        f"{research_brief}\n\n"
        f"{FALLBACK_PLAN_CONFIRMATION_FOOTER}"
    )


async def _handle_clarification_or_plan(
    state: ResearchState,
    messages: list[Any],
    config: RunnableConfig | None,
) -> Command[Literal["research_supervisor", "__end__"]]:
    follow_up_topic_shift = _is_follow_up_topic_shift(state, messages)

    latest_user_text = latest_human_text(messages)
    if not latest_user_text:
        return _missing_user_text_command()

    if _needs_scope_boundary_clarification(messages):
        return _scope_boundary_clarification_command(follow_up_topic_shift=follow_up_topic_shift)

    response = await _invoke_clarification_check(messages, config)
    if bool(getattr(response, "need_clarification", False)):
        question = str(getattr(response, "question", "")).strip() or FALLBACK_CLARIFY_QUESTION
        return _clarification_command(question, follow_up_topic_shift=follow_up_topic_shift)

    research_brief = await _generate_research_brief(messages, config)
    if is_broad_scope_request(messages):
        plan = await _generate_research_plan(research_brief, config)
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=_plan_message(plan, research_brief))],
                "research_brief": research_brief,
                "intake_decision": "clarify",
                "awaiting_clarification": True,
            },
        )

    verification = str(getattr(response, "verification", "")).strip() or FALLBACK_VERIFICATION
    update = _build_research_handoff_update(research_brief)
    update["messages"] = [AIMessage(content=verification)]
    return Command(goto="research_supervisor", update=update)


async def scope_intake(
    state: ResearchState,
    config: RunnableConfig = None,  # type: ignore[assignment]
) -> Command[Literal["research_supervisor", "__end__"]]:
    """Handle clarification and brief synthesis before handing off to the supervisor."""
    messages = _state_messages(state)
    if _is_plan_acknowledgement_turn(state, messages):
        return await _handle_plan_acknowledgement(messages, config)
    if not _should_run_clarification_pass(state, messages):
        return await _handle_direct_proceed(messages, config)
    return await _handle_clarification_or_plan(state, messages, config)
