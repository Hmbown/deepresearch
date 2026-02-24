"""Researcher subgraph backed by a deep agent with built-in middleware."""

from __future__ import annotations

import re
from typing import Any

from .config import (
    get_max_evidence_claims_per_research_unit,
    get_max_source_urls_per_claim,
    get_max_react_tool_calls,
    get_model_string,
    get_researcher_search_budget,
    get_search_tool,
)
from .nodes import _build_fetch_url_tool, _build_search_tool_with_processing, think_tool
from .prompts import RESEARCHER_PROMPT, RESEARCHER_PROMPT_NO_SEARCH
from .state import EvidenceRecord, compress_note_text, stringify_tool_output

try:  # pragma: no cover - import is environment-dependent
    from deepagents import create_deep_agent as _deepagents_create_deep_agent
except ImportError:  # pragma: no cover - defensive fallback for environments without deepagents
    _deepagents_create_deep_agent = None

# Kept as a module attribute so tests can monkeypatch it directly.
create_deep_agent = _deepagents_create_deep_agent

_URL_PATTERN = re.compile(r"https?://[^\s<>\]\"')]+")
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CITATION_URL_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?\[(\d+)\]:?\s+(https?://\S+)"
)
_SOURCE_HEADER_PATTERN = re.compile(r"(?i)^\s*(?:#{1,6}\s*)?(?:sources?|references?)\s*:?\s*$")
_SECTION_HEADER_PATTERN = re.compile(
    r"(?i)^\s*(?:#{1,6}\s*)?(?:executive summary|key findings|evidence log|"
    r"contradictions/uncertainties|gaps/next questions)\s*:?\s*$"
)
_UNCERTAINTY_PATTERN = re.compile(
    r"(?i)\b(?:uncertain|uncertainty|contradict|mixed evidence|inconclusive|"
    r"not clear|unknown|limited evidence|weak evidence|disputed)\b"
)


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
    prompt_values: dict[str, Any] = {
        "max_react_tool_calls": get_max_react_tool_calls(),
    }
    if search_enabled:
        prompt_values["researcher_search_budget"] = get_researcher_search_budget()
    return prompt_template.format(**prompt_values)


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


def _extract_citation_url_map(text: str) -> dict[str, str]:
    citation_map: dict[str, str] = {}
    for line in text.splitlines():
        match = _CITATION_URL_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        citation_map[match.group(1)] = _normalize_url(match.group(2))
    return citation_map


def _extract_claim_lines(text: str, max_claims: int) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()
    in_source_section = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _SOURCE_HEADER_PATTERN.match(line):
            in_source_section = True
            continue

        if _SECTION_HEADER_PATTERN.match(line):
            continue

        if in_source_section:
            if (
                _CITATION_URL_LINE_PATTERN.match(line)
                or line.lower().startswith("url:")
                or _URL_PATTERN.match(line)
                or re.match(r"^\s*[-*]\s*https?://", line)
            ):
                continue
            in_source_section = False

        cleaned = re.sub(r"^[*-]\s*", "", line)
        if len(cleaned) < 20:
            continue

        dedupe_key = re.sub(r"\s+", " ", cleaned).strip().lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        claims.append(cleaned)
        if len(claims) >= max_claims:
            break

    return claims


def _extract_evidence_records(raw_text: str) -> list[EvidenceRecord]:
    text = raw_text.strip()
    if not text:
        return []

    citation_map = _extract_citation_url_map(text)
    all_urls = [_normalize_url(url) for url in _URL_PATTERN.findall(text)]
    deduped_all_urls: list[str] = []
    for url in all_urls:
        if url and url not in deduped_all_urls:
            deduped_all_urls.append(url)

    max_claims_per_unit = get_max_evidence_claims_per_research_unit()
    claim_lines = _extract_claim_lines(text, max_claims=max_claims_per_unit)
    if not claim_lines:
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        if first_sentence:
            claim_lines = [first_sentence[:280]]

    records: list[EvidenceRecord] = []
    max_urls_per_claim = get_max_source_urls_per_claim()
    for claim in claim_lines:
        urls: list[str] = []
        for citation_id in _CITATION_PATTERN.findall(claim):
            mapped_url = citation_map.get(citation_id)
            if mapped_url and mapped_url not in urls:
                urls.append(mapped_url)
            if len(urls) >= max_urls_per_claim:
                break
        for url in _URL_PATTERN.findall(claim):
            normalized = _normalize_url(url)
            if normalized and normalized not in urls:
                urls.append(normalized)
            if len(urls) >= max_urls_per_claim:
                break
        # Fallback: if a claim has citation markers that resolved to nothing and
        # there are few enough global URLs to be meaningful, assign them.
        if not urls:
            if len(deduped_all_urls) == 1:
                urls = [deduped_all_urls[0]]
            elif _CITATION_PATTERN.search(claim) and len(deduped_all_urls) <= 5:
                urls = list(deduped_all_urls)
            if len(urls) > max_urls_per_claim:
                urls = urls[:max_urls_per_claim]

        uncertainty = claim if _UNCERTAINTY_PATTERN.search(claim) else None
        records.append(
            EvidenceRecord(
                claim=claim,
                source_urls=urls,
                # Keep confidence deterministic and model-agnostic.
                confidence=0.5,
                contradiction_or_uncertainty=uncertainty,
            )
        )

    return records


def extract_research_from_messages(result: dict) -> tuple[str | None, list[str], list[EvidenceRecord]]:
    """Post-process deep agent MessagesState into compressed notes + raw notes + evidence."""
    messages = result.get("messages", [])
    note_lines: list[str] = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type not in {"ai", "tool"}:
            continue
        if msg_type == "tool" and getattr(msg, "name", "") == think_tool.name:
            continue
        line = stringify_tool_output(getattr(msg, "content", "")).strip()
        if line:
            note_lines.append(line)

    raw_text = "\n\n".join(note_lines).strip()
    compressed = compress_note_text(raw_text)
    raw_notes = [raw_text] if raw_text else []
    evidence_ledger = _extract_evidence_records(raw_text)
    return compressed or raw_text or None, raw_notes, evidence_ledger
