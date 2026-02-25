"""Run a real two-turn runtime validation for clarify -> proceed -> 3+ fanout."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

from deepresearch import cli
from deepresearch.env import bootstrap_env

TURN_1_QUERY = "Please research this topic for me."
TURN_2_QUERY = (
    "I need a concrete comparative brief for 2026 planning. "
    "Important process requirement: do not finalize until you have run three independent research tracks, "
    "then synthesize them together. "
    "Topic: whether U.S. grid operators should prioritize utility-scale batteries, advanced nuclear, "
    "or demand-response programs through 2035. "
    "Track A: cost curves and deployment timelines. "
    "Track B: reliability impacts and grid-integration risks. "
    "Track C: policy and regulatory blockers in the U.S. and EU. "
    "In the final report, include separate findings for Track A, Track B, and Track C with citations."
)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _langsmith_enabled() -> bool:
    return _is_truthy(os.environ.get("LANGCHAIN_TRACING_V2")) or _is_truthy(os.environ.get("LANGSMITH_TRACING"))


def _langsmith_project() -> str:
    return os.environ.get("LANGCHAIN_PROJECT") or os.environ.get("LANGSMITH_PROJECT") or "deepresearch-local"


def _count_conduct_research_messages(messages: list[Any]) -> int:
    count = 0
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        if getattr(message, "name", "") != "ConductResearch":
            continue

        content = str(getattr(message, "content", ""))
        if (
            "[ConductResearch skipped:" in content
            or "[ConductResearch missing" in content
            or "[Research unit failed:" in content
        ):
            continue
        count += 1
    return count


def _verify_langsmith_trace(max_wait_seconds: int = 30) -> tuple[bool, str]:
    project_name = _langsmith_project()
    try:
        from langsmith import Client

        client = Client()
    except Exception as exc:  # pragma: no cover - depends on runtime/env
        return False, f"LangSmith client init failed: {exc}"

    deadline = time.time() + max_wait_seconds
    last_count = 0
    while time.time() < deadline:
        try:
            runs = list(client.list_runs(project_name=project_name, limit=20))
        except Exception as exc:  # pragma: no cover - network/provider dependent
            return False, f"LangSmith list_runs failed: {exc}"
        last_count = len(runs)
        if runs:
            return True, f"Detected {len(runs)} run(s) in project `{project_name}`."
        time.sleep(5)

    return False, f"No runs detected in project `{project_name}` within {max_wait_seconds}s (count={last_count})."


async def _run_validation() -> int:
    bootstrap_env()

    # Force the minimum fanout target requested by release validation.
    os.environ["MAX_CONCURRENT_RESEARCH_UNITS"] = "3"
    os.environ["MAX_RESEARCHER_ITERATIONS"] = "3"

    thread_id = f"release-validation-{uuid.uuid4().hex}"
    turn_1_result = await cli.run(TURN_1_QUERY, thread_id=thread_id)
    turn_1_decision = str(turn_1_result.get("intake_decision") or "")
    turn_1_text = cli._final_assistant_text(turn_1_result).strip()

    prior_messages = list(turn_1_result.get("messages", []))
    turn_2_result = await cli.run(
        TURN_2_QUERY,
        thread_id=thread_id,
        prior_messages=prior_messages,
    )
    turn_2_decision = str(turn_2_result.get("intake_decision") or "")
    conduct_research_count = _count_conduct_research_messages(list(turn_2_result.get("supervisor_messages", [])))

    final_report = str(turn_2_result.get("final_report") or "").strip() or cli._final_assistant_text(turn_2_result)
    final_report_preview = final_report[:300]

    print("=== Long Multi-Agent Validation ===")
    print(f"thread_id={thread_id}")
    print(f"turn1.decision={turn_1_decision}")
    print(f"turn1.clarification={turn_1_text}")
    print(f"turn2.decision={turn_2_decision}")
    print(f"turn2.conduct_research_count={conduct_research_count}")
    print(f"turn2.final_report_length={len(final_report)}")
    print(f"turn2.final_report_first_300={final_report_preview}")

    validation_errors: list[str] = []
    if turn_1_decision != "clarify":
        validation_errors.append(f"Expected turn1 intake_decision=clarify, got `{turn_1_decision}`.")
    if turn_2_decision != "proceed":
        validation_errors.append(f"Expected turn2 intake_decision=proceed, got `{turn_2_decision}`.")
    if conduct_research_count < 3:
        validation_errors.append(f"Expected >=3 executed ConductResearch messages, got {conduct_research_count}.")

    if _langsmith_enabled():
        langsmith_ok, langsmith_message = _verify_langsmith_trace(max_wait_seconds=30)
        print("langsmith.enabled=true")
        print(f"langsmith.trace_check={'PASS' if langsmith_ok else 'FAIL'}")
        print(f"langsmith.detail={langsmith_message}")
        if not langsmith_ok:
            validation_errors.append(langsmith_message)
    else:
        print("langsmith.enabled=false")
        print("langsmith.trace_check=SKIP")
        print("langsmith.detail=Tracing disabled in environment.")

    if validation_errors:
        print("validation_result=FAIL")
        for error in validation_errors:
            print(f"validation_error={error}")
        return 1

    print("validation_result=PASS")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run_validation()))


if __name__ == "__main__":
    main()
