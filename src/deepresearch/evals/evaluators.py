"""LLM-as-judge evaluators for answer quality, process quality, and composite scoring."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from langsmith.schemas import Run

from ..config import (
    get_eval_model,
    get_max_concurrent_research_units,
    get_max_react_tool_calls,
    get_max_researcher_iterations,
    get_researcher_search_budget,
)
from .prompts import ANSWER_QUALITY_PROMPT, PROCESS_QUALITY_PROMPT


def _extract_score(text: str) -> float | None:
    """Extract a 0-1 float score from judge output text.

    Looks for patterns like "0.85", "Score: 0.7", "final score: 0.65" etc.
    Returns the last match (judges typically reason first, score last).
    """
    matches = re.findall(r"(?:^|[\s:])([01](?:\.\d+)?)\s*$", text, re.MULTILINE)
    if not matches:
        matches = re.findall(r"\b([01]\.\d+)\b", text)
    if not matches:
        return None
    val = float(matches[-1])
    return max(0.0, min(1.0, val))


def _get_final_report(run: Run) -> str:
    """Extract the final report text from a root run's outputs."""
    outputs = run.outputs or {}
    final_report = outputs.get("final_report", "")
    if isinstance(final_report, str) and final_report.strip():
        return final_report.strip()

    messages = outputs.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("type") == "ai":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    return "\n".join(parts).strip()
        elif hasattr(msg, "type") and msg.type == "ai":
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content.strip()
    return ""


def _get_user_query(run: Run) -> str:
    """Extract the original user query from a root run's inputs."""
    inputs = run.inputs or {}
    messages = inputs.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("type") in ("human", "user"):
                content = msg.get("content", "")
                return content if isinstance(content, str) else str(content)
        elif hasattr(msg, "type") and msg.type == "human":
            return str(getattr(msg, "content", ""))
    return ""


def _run_judge(prompt_template: str, inputs: str, outputs: str) -> tuple[float | None, str]:
    """Run an LLM-as-judge evaluation and return (score, reasoning).

    Uses openevals create_llm_as_judge for structured scoring.
    Falls back to direct model invocation if openevals is unavailable.
    """
    from openevals.llm import create_llm_as_judge

    judge = create_llm_as_judge(
        prompt=prompt_template,
        model=get_eval_model(),
        continuous=True,
    )
    result = judge(inputs=inputs, outputs=outputs)
    score = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
    reasoning = ""
    if isinstance(result, dict):
        reasoning = result.get("comment", "") or result.get("reasoning", "")
    else:
        reasoning = getattr(result, "comment", "") or getattr(result, "reasoning", "")

    if score is not None:
        score = max(0.0, min(1.0, float(score)))
    return score, reasoning


def eval_answer_quality(run: Run, example: Any = None) -> dict[str, Any]:
    """Evaluate the quality of the final research report.

    Compatible with LangSmith RunEvaluator interface.
    Returns dict with key, score, and comment.
    """
    report = _get_final_report(run)
    if not report:
        return {"key": "answer_quality", "score": 0.0, "comment": "No final report found in run outputs."}

    query = _get_user_query(run)
    inputs_text = query or "(no user query found)"

    score, reasoning = _run_judge(ANSWER_QUALITY_PROMPT, inputs=inputs_text, outputs=report)
    if score is None:
        score = _extract_score(reasoning) if reasoning else None
    return {
        "key": "answer_quality",
        "score": score,
        "comment": reasoning,
    }


def _build_process_summary(child_runs: list[Any]) -> str:
    """Build a text summary of the research process from child runs."""
    conduct_research_count = 0
    search_web_count = 0
    fetch_url_count = 0
    think_tool_count = 0
    source_domains: set[str] = set()
    observed_names: set[str] = set()

    def _normalize_run_name(raw_name: Any) -> str:
        if not isinstance(raw_name, str):
            return ""
        return raw_name.rsplit("/", 1)[-1].rsplit(":", 1)[-1].rsplit(".", 1)[-1]

    def _extract_domain(url: str) -> str | None:
        try:
            parsed = urlparse(url.strip())
        except Exception:
            return None
        if not parsed.netloc:
            return None
        return parsed.netloc

    for child in child_runs:
        raw_name = getattr(child, "name", "")
        if not raw_name and isinstance(child, dict):
            raw_name = child.get("name", "")
        name = _normalize_run_name(raw_name)
        if name:
            observed_names.add(name)
        inputs = getattr(child, "inputs", {}) or {}
        outputs = getattr(child, "outputs", {}) or {}

        if name == "ConductResearch":
            content = ""
            if isinstance(outputs, dict):
                content = str(outputs.get("output", "") or outputs.get("content", ""))
            elif isinstance(outputs, str):
                content = outputs
            if "[ConductResearch skipped:" not in content and "[Research unit failed:" not in content:
                conduct_research_count += 1

        if name == "search_web":
            search_web_count += 1
            if isinstance(outputs, str):
                results = outputs
            elif isinstance(outputs, list):
                results = outputs
            elif isinstance(outputs, dict):
                results = outputs.get("results", [])
            else:
                results = []
            if isinstance(results, str):
                urls = re.findall(r"https?://[^\s\]\"']+", results)
                for url in urls:
                    domain = _extract_domain(url)
                    if domain:
                        source_domains.add(domain)
            elif isinstance(results, list):
                for item in results:
                    url = item.get("url", "") if isinstance(item, dict) else ""
                    if url:
                        domain = _extract_domain(url)
                        if domain:
                            source_domains.add(domain)

        if name == "fetch_url":
            fetch_url_count += 1
            url = ""
            if isinstance(inputs, dict):
                url = str(inputs.get("url", "") or inputs.get("input", ""))
            if url:
                domain = _extract_domain(url)
                if domain:
                    source_domains.add(domain)

        if name == "think_tool":
            think_tool_count += 1

    sorted_domains = sorted(source_domains)
    domains_str = ", ".join(sorted_domains[:20]) if sorted_domains else "(none)"
    config_lines = (
        f"- Runtime config: max_react_tool_calls={get_max_react_tool_calls()}, "
        f"search_budget={get_researcher_search_budget()}, "
        f"max_concurrent_research_units={get_max_concurrent_research_units()}, "
        f"max_researcher_iterations={get_max_researcher_iterations()}\n"
        f"- Observed child run names: {', '.join(sorted(observed_names)) or '(none)'}\n"
    )

    return (
        f"Research Process Summary:\n"
        f"- ConductResearch units dispatched: {conduct_research_count}\n"
        f"- search_web calls: {search_web_count}\n"
        f"- fetch_url calls: {fetch_url_count}\n"
        f"- think_tool calls: {think_tool_count}\n"
        f"{config_lines}"
        f"- Unique source domains ({len(source_domains)}): {domains_str}\n"
    )


def eval_process_quality(run: Run, client: Any) -> dict[str, Any]:
    """Evaluate research process quality by inspecting child runs.

    Requires a LangSmith client to fetch the run tree.
    Returns dict with key, score, and comment.
    """
    trace_id = run.trace_id or run.id
    try:
        child_runs = list(client.list_runs(trace_id=trace_id))
    except Exception as exc:
        return {
            "key": "process_quality",
            "score": None,
            "comment": f"Failed to fetch child runs: {exc}",
        }

    summary = _build_process_summary(child_runs)
    score, reasoning = _run_judge(PROCESS_QUALITY_PROMPT, inputs="", outputs=summary)
    if score is None:
        score = _extract_score(reasoning) if reasoning else None
    return {
        "key": "process_quality",
        "score": score,
        "comment": f"{summary}\n---\n{reasoning}",
    }


def eval_composite(run: Run, client: Any) -> dict[str, Any]:
    """Run both evaluators and compute a weighted composite score.

    Composite = 0.6 * answer_quality + 0.4 * process_quality.
    Returns dict with key, score, and comment including breakdown.
    """
    answer_result = eval_answer_quality(run)
    process_result = eval_process_quality(run, client)

    answer_score = answer_result.get("score")
    process_score = process_result.get("score")

    composite = None
    if answer_score is not None and process_score is not None:
        composite = round(0.6 * answer_score + 0.4 * process_score, 4)

    breakdown = (
        f"answer_quality={answer_score}, "
        f"process_quality={process_score}, "
        f"composite={composite} (0.6*answer + 0.4*process)"
    )
    return {
        "key": "composite_quality",
        "score": composite,
        "comment": breakdown,
        "answer_result": answer_result,
        "process_result": process_result,
    }
