"""Researcher subgraph backed by a deep agent with built-in middleware."""

from __future__ import annotations

import re
from typing import Any

from .config import (
    get_max_react_tool_calls,
    get_model_string,
    get_search_tool,
)
from .nodes import _build_fetch_url_tool, _build_search_tool_with_processing, think_tool
from .prompts import RESEARCHER_PROMPT, RESEARCHER_PROMPT_NO_SEARCH
from .state import EvidenceRecord, state_text_or_none, stringify_tool_output

try:  # pragma: no cover - import is environment-dependent
    from deepagents import create_deep_agent as _deepagents_create_deep_agent
except ImportError:  # pragma: no cover - defensive fallback for environments without deepagents
    _deepagents_create_deep_agent = None

# Kept as a module attribute so tests can monkeypatch it directly.
create_deep_agent = _deepagents_create_deep_agent

_URL_PATTERN = re.compile(r"https?://[^\s<>\]\"')]+")



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
) -> tuple[list[Any], bool]:
    base_search_tool = get_search_tool()
    fetch_url_tool = _build_fetch_url_tool(writer)
    tools = [think_tool, fetch_url_tool]
    if base_search_tool is not None:
        tools.insert(0, _build_search_tool_with_processing(base_search_tool, writer))
    return tools, base_search_tool is not None


def render_researcher_prompt(*, search_enabled: bool = True) -> str:
    prompt_template = RESEARCHER_PROMPT if search_enabled else RESEARCHER_PROMPT_NO_SEARCH
    return prompt_template.format(max_react_tool_calls=get_max_react_tool_calls())


def build_researcher_subgraph():
    """Build a deep-agent researcher with built-in middleware."""
    model_str = get_model_string("subagent")
    tools, search_enabled = _build_research_tools_and_capabilities()
    system_prompt = render_researcher_prompt(search_enabled=search_enabled)
    create_agent = _resolve_create_deep_agent()

    return create_agent(
        model=model_str,
        tools=tools,
        system_prompt=system_prompt,
        name="deep-researcher",
    )


def _normalize_url(url: str) -> str:
    return url.strip().rstrip(".,;")


def _extract_evidence_records(raw_text: str) -> list[EvidenceRecord]:
    """Extract one EvidenceRecord per unique URL found in researcher output."""
    text = raw_text.strip()
    if not text:
        return []

    seen: set[str] = set()
    records: list[EvidenceRecord] = []
    for raw_url in _URL_PATTERN.findall(text):
        url = _normalize_url(raw_url)
        if not url or url in seen:
            continue
        seen.add(url)
        records.append(
            EvidenceRecord(
                source_urls=[url],
            )
        )
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

    evidence_ledger = _extract_evidence_records(raw_text)
    if not evidence_ledger:
        # Fallback: if the researcher forgot to include sources in the final write-up,
        # recover URLs from tool outputs without retaining their full content.
        url_lines: list[str] = []
        seen_urls: set[str] = set()
        for msg in messages:
            if getattr(msg, "type", "") != "tool":
                continue
            if getattr(msg, "name", "") == think_tool.name:
                continue
            tool_text = stringify_tool_output(getattr(msg, "content", "")).strip()
            if not tool_text:
                continue
            for raw_url in _URL_PATTERN.findall(tool_text):
                url = _normalize_url(raw_url)
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                url_lines.append(url)

        if url_lines:
            evidence_ledger = [EvidenceRecord(source_urls=[url]) for url in url_lines]

    raw_notes = [raw_text] if raw_text else []
    # Treat the researcher's final write-up as the "compressed" artifact for downstream use.
    # (Avoid lossy bulletization that can destroy structure/citations.)
    compressed = state_text_or_none(raw_text)
    return compressed, raw_notes, evidence_ledger
