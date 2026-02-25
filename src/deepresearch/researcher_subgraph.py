"""Researcher subgraph backed by a deep agent with built-in middleware."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from .config import (
    get_max_react_tool_calls,
    get_llm,
    get_search_tool,
)
from .nodes import _build_fetch_url_tool, _build_search_tool_with_processing, think_tool
from .prompts import RESEARCHER_PROMPT
from .state import EvidenceRecord, EvidenceSourceType, state_text_or_none, stringify_tool_output

try:  # pragma: no cover - import is environment-dependent
    from deepagents import create_deep_agent as _deepagents_create_deep_agent
except ImportError:  # pragma: no cover - defensive fallback for environments without deepagents
    _deepagents_create_deep_agent = None

# Kept as a module attribute so tests can monkeypatch it directly.
create_deep_agent = _deepagents_create_deep_agent

_URL_PATTERN = re.compile(r"https?://[^\s<>\]\"')]+")
_SEARCH_URL_LINE_PATTERN = re.compile(r"(?im)^\s*URL:\s*(\S+)")



def _resolve_create_deep_agent():
    if create_deep_agent is not None:
        return create_deep_agent

    try:
        from deepagents import create_deep_agent as runtime_create_deep_agent
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "deepagents is required to build the researcher subgraph. "
            "Install project dependencies (for example `pip install -e .`)."
        ) from exc

    return runtime_create_deep_agent


def _build_research_tools_and_capabilities(
    writer: Any | None = None,
) -> list[Any]:
    base_search_tool = get_search_tool()
    fetch_url_tool = _build_fetch_url_tool(writer)
    tools = [think_tool, fetch_url_tool]
    if base_search_tool is not None:
        tools.insert(0, _build_search_tool_with_processing(base_search_tool, writer))
    return tools


def render_researcher_prompt() -> str:
    return RESEARCHER_PROMPT.format(max_react_tool_calls=get_max_react_tool_calls())


def build_researcher_subgraph():
    """Build a deep-agent researcher with built-in middleware."""
    model = get_llm("subagent", prefer_compact_context=True)
    tools = _build_research_tools_and_capabilities()
    system_prompt = render_researcher_prompt()
    create_agent = _resolve_create_deep_agent()

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        name="deep-researcher",
    )


def _normalize_url(url: str) -> str:
    return url.strip().rstrip(".,;")


def _is_http_url(url: str) -> bool:
    normalized = _normalize_url(url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _extract_evidence_records(
    raw_text: str,
    *,
    source_type: EvidenceSourceType = "fetched",
) -> list[EvidenceRecord]:
    """Extract one EvidenceRecord per unique URL found in researcher output."""
    text = raw_text.strip()
    if not text:
        return []

    seen: set[str] = set()
    records: list[EvidenceRecord] = []
    for raw_url in _URL_PATTERN.findall(text):
        url = _normalize_url(raw_url)
        if not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        records.append(
            EvidenceRecord(
                source_urls=[url],
                source_type=source_type,
            )
        )
    return records


def _tool_call_lookup(messages: list[Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if getattr(msg, "type", "") != "ai":
            continue
        raw_tool_calls = getattr(msg, "tool_calls", None)
        if not isinstance(raw_tool_calls, list):
            continue
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            call_id = str(raw_call.get("id") or "").strip()
            if not call_id:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            lookup[call_id] = {
                "name": str(raw_call.get("name") or "").strip(),
                "args": args,
            }
    return lookup


def _extract_fetched_evidence_from_messages(messages: list[Any]) -> list[EvidenceRecord]:
    """Extract URLs that came from actual search/fetch tool usage."""
    call_lookup = _tool_call_lookup(messages)
    seen_urls: set[str] = set()
    records: list[EvidenceRecord] = []

    def add_url(raw_url: str) -> None:
        url = _normalize_url(raw_url)
        if not _is_http_url(url) or url in seen_urls:
            return
        seen_urls.add(url)
        records.append(EvidenceRecord(source_urls=[url], source_type="fetched"))

    for msg in messages:
        if getattr(msg, "type", "") != "tool":
            continue

        call_id = str(getattr(msg, "tool_call_id", "") or "").strip()
        call = call_lookup.get(call_id, {})
        tool_name = str(getattr(msg, "name", "") or call.get("name") or "").strip()
        if tool_name == think_tool.name:
            continue

        tool_text = stringify_tool_output(getattr(msg, "content", "")).strip()
        if tool_name == "fetch_url":
            fetch_arg_url = str(call.get("args", {}).get("url") or "").strip()
            if fetch_arg_url:
                add_url(fetch_arg_url)
                continue
            if not tool_text:
                continue
            for raw_url in _URL_PATTERN.findall(tool_text):
                add_url(raw_url)
            continue

        if tool_name == "search_web":
            candidate_urls = _SEARCH_URL_LINE_PATTERN.findall(tool_text) if tool_text else []
            if not candidate_urls and tool_text:
                candidate_urls = _URL_PATTERN.findall(tool_text)
            for raw_url in candidate_urls:
                add_url(raw_url)
            continue

        # Defensive fallback for any future URL-returning research tools.
        if not tool_text:
            continue
        for raw_url in _URL_PATTERN.findall(tool_text):
            add_url(raw_url)

    return records


def extract_research_from_messages(result: dict) -> tuple[str | None, list[str], list[EvidenceRecord]]:
    """Post-process deep agent MessagesState into compressed notes + raw notes + evidence."""
    raw_messages = result.get("messages", [])
    messages = raw_messages if isinstance(raw_messages, list) else []

    # The deep agent returns a full tool trace (search results, fetched page bodies, etc.)
    # but downstream supervisor/report prompts should only see the researcher's synthesized
    # write-up. Tool outputs can be extremely token-heavy (e.g., SEC filings), so we avoid
    # folding them into notes.
    ai_chunks: list[str] = []
    for msg in messages:
        if getattr(msg, "type", "") != "ai":
            continue
        content = stringify_tool_output(getattr(msg, "content", "")).strip()
        if content:
            ai_chunks.append(content)

    raw_text = ai_chunks[-1].strip() if ai_chunks else ""
    fetched_evidence = _extract_fetched_evidence_from_messages(messages)
    model_cited_evidence = _extract_evidence_records(raw_text, source_type="model_cited")
    evidence_ledger = list(fetched_evidence)

    seen_urls = {
        url
        for record in fetched_evidence
        for url in record.source_urls
        if _is_http_url(url)
    }
    for record in model_cited_evidence:
        urls = [url for url in record.source_urls if _is_http_url(url)]
        if not urls:
            continue
        primary_url = urls[0]
        if primary_url in seen_urls:
            continue
        seen_urls.add(primary_url)
        evidence_ledger.append(record)

    raw_notes = [raw_text] if raw_text else []
    # Treat the researcher's final write-up as the "compressed" artifact for downstream use.
    # (Avoid lossy bulletization that can destroy structure/citations.)
    compressed = state_text_or_none(raw_text)
    return compressed, raw_notes, evidence_ledger
