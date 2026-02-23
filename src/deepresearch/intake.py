"""Intake routing and brief generation nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from .config import get_llm, get_max_structured_output_retries
from .prompts import CLARIFY_PROMPT, RESEARCH_BRIEF_PROMPT
from .runtime_utils import invoke_structured_with_retries
from .state import (
    FALLBACK_CLARIFY_QUESTION,
    FALLBACK_FOLLOWUP_CLARIFICATION,
    FALLBACK_VERIFICATION,
    ClarifyWithUser,
    ResearchBrief,
    ResearchState,
    latest_human_text,
    should_recheck_intent_on_follow_up,
    today_utc_date,
)


async def route_turn(
    state: ResearchState,
    config: RunnableConfig = None,
) -> Command[Literal["clarify_with_user", "write_research_brief"]]:
    """Route each turn to intake or direct brief generation."""
    del config  # RunnableConfig is accepted for compatibility with graph invocation.
    if bool(state.get("awaiting_clarification")):
        return Command(goto="clarify_with_user")
    if state.get("intake_decision") is None and not state.get("research_brief"):
        return Command(goto="clarify_with_user")
    if state.get("intake_decision") == "proceed" and should_recheck_intent_on_follow_up(
        list(convert_to_messages(state.get("messages", [])))
    ):
        return Command(goto="clarify_with_user")
    return Command(goto="write_research_brief")


async def clarify_with_user(
    state: ResearchState,
    config: RunnableConfig = None,
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Ask one clarification question when needed; otherwise acknowledge and proceed."""
    messages = list(convert_to_messages(state.get("messages", [])))
    latest_user_text = latest_human_text(messages)
    if not latest_user_text:
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=FALLBACK_CLARIFY_QUESTION)],
                "intake_decision": "clarify",
                "awaiting_clarification": True,
            },
        )

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
        response = SimpleNamespace(
            need_clarification=True,
            question=FALLBACK_FOLLOWUP_CLARIFICATION,
            verification="",
        )

    need_clarification = bool(getattr(response, "need_clarification", False))
    if need_clarification:
        question = str(getattr(response, "question", "")).strip() or FALLBACK_CLARIFY_QUESTION
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=question)],
                "intake_decision": "clarify",
                "awaiting_clarification": True,
            },
        )

    verification = str(getattr(response, "verification", "")).strip() or FALLBACK_VERIFICATION
    return Command(
        goto="write_research_brief",
        update={
            "messages": [AIMessage(content=verification)],
            "intake_decision": "proceed",
            "awaiting_clarification": False,
        },
    )


async def write_research_brief(state: ResearchState, config: RunnableConfig = None) -> dict[str, Any]:
    """Generate a focused research brief from full chat history."""
    messages = list(convert_to_messages(state.get("messages", [])))
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

    supervisor_messages = [HumanMessage(content=research_brief)] if research_brief else []
    return {
        "research_brief": research_brief,
        "supervisor_messages": supervisor_messages,
        "notes": [],
        "raw_notes": [],
        "final_report": "",
        "awaiting_clarification": False,
    }
