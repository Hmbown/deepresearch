"""Deep Research Agent CLI entry point."""

from __future__ import annotations

import asyncio
import sys
from typing import Any
import uuid

from langchain_core.messages import HumanMessage

from .config import online_evals_enabled
from .env import ensure_runtime_env_ready, runtime_preflight
from .message_utils import extract_text_content as _extract_text_content


def _final_assistant_text(result: dict[str, Any]) -> str:
    final_report = _extract_text_content(result.get("final_report", "")).strip()
    if final_report:
        return final_report

    messages = result.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            return _extract_text_content(getattr(message, "content", ""))
    return ""


def _get_app():
    from .graph import app

    return app


def _new_thread_id() -> str:
    return uuid.uuid4().hex


def _thread_config(thread_id: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if online_evals_enabled():
        from .evals import attach_online_eval_callback

        cfg = attach_online_eval_callback(cfg)
    return cfg


async def run(
    query: str,
    thread_id: str | None = None,
    prior_messages: list[Any] | None = None,
) -> dict[str, Any]:
    """Run a deep research query and return the agent result state."""
    ensure_runtime_env_ready()
    resolved_thread_id = (thread_id or "").strip() or _new_thread_id()
    payload_messages = list(prior_messages or [])
    payload_messages.append(HumanMessage(content=query))
    return await _get_app().ainvoke(
        {"messages": payload_messages},
        config=_thread_config(resolved_thread_id),
    )


def _result_section_title(result: dict[str, Any]) -> str:
    return "CLARIFICATION" if result.get("intake_decision") == "clarify" else "RESPONSE"


def print_results(result: dict[str, Any]) -> None:
    """Print the final assistant response."""
    response_text = _final_assistant_text(result)
    if response_text:
        section_title = _result_section_title(result)
        print("\n" + "=" * 70)
        print(section_title)
        print("=" * 70)
        print(response_text)
    else:
        print("No assistant response found.")


def print_preflight(project_name: str | None = None) -> int:
    """Run setup preflight checks and print status lines."""
    ok, checks = runtime_preflight(project_name=project_name)
    print("Deep Research Preflight")
    print("-" * 40)
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.message}")
    return 0 if ok else 1


async def run_session(thread_id: str) -> None:
    """Run an interactive multi-turn session on a single thread."""
    print(f"Session thread_id: {thread_id}")
    print("Type 'exit' or 'quit' to end the session.")
    prior_messages: list[Any] = []
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q", "/exit"}:
            return
        result = await run(query, thread_id=thread_id, prior_messages=prior_messages)
        prior_messages = list(result.get("messages", prior_messages))
        print_results(result)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--preflight":
        project_name = args[1].strip() if len(args) > 1 else None
        raise SystemExit(print_preflight(project_name=project_name or None))

    if args:
        query = " ".join(args).strip()
        if not query:
            print("No query provided.")
            raise SystemExit(1)
        thread_id = _new_thread_id()
        print(f"\nResearching: {query}\n")
        result = asyncio.run(run(query, thread_id=thread_id))
        print_results(result)
    else:
        print("Deep Research Agent")
        print("-" * 40)
        provided_thread = input("Thread ID (optional, press enter to auto-generate): ").strip()
        thread_id = provided_thread or _new_thread_id()
        asyncio.run(run_session(thread_id))


if __name__ == "__main__":
    main()
