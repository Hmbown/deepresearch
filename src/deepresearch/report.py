"""Final report synthesis node."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from .config import get_llm
from .prompts import FINAL_REPORT_PROMPT
from .runtime_utils import invoke_runnable_with_config, log_runtime_event
from .state import (
    FALLBACK_FINAL_REPORT,
    EvidenceRecord,
    ResearchState,
    compress_note_text,
    is_token_limit_error,
    join_note_list,
    normalize_evidence_ledger,
    normalize_note_list,
    state_text_or_none,
    stringify_tool_output,
    today_utc_date,
)

_logger = logging.getLogger(__name__)

_SOURCE_URL_PATTERN = re.compile(r"https?://[^\s<>\]\"')]+")
_SOURCE_SECTION_HEADER_PATTERN = re.compile(
    r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:sources?|references?)\s*:?\s*$"
)
_NO_SOURCE_URLS_SENTINEL_PATTERN = re.compile(
    r"(?im)^\s*[-*]?\s*No source URLs were available in collected notes\.?\s*$"
)
_INTERNAL_META_LINE_PATTERNS = (
    re.compile(r"(?i)^\s*\[(?:ConductResearch|ResearchComplete|Research unit failed|ConductResearch skipped).*"),
    re.compile(r"(?i)^\s*Reflection recorded:"),
    re.compile(r"(?i)^\s*(?:Raw|Compressed) notes so far\s*:"),
    re.compile(r"(?i)\b(?:tool_call_id|supervisor_research_|supervisor_think_)\b"),
    re.compile(
        r"(?i)\b(?:i|we)\s+(?:used|called|ran|invoked)\s+(?:the\s+)?"
        r"(?:ConductResearch|ResearchComplete|search_web|fetch_url|think_tool)\b"
    ),
)


def _normalize_source_url(raw_url: str) -> str:
    return str(raw_url).strip().rstrip(".,;")


def _is_valid_source_url(raw_url: str) -> bool:
    normalized = _normalize_source_url(raw_url)
    if not normalized or normalized.endswith("-"):
        return False
    parsed = urlparse(normalized)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _extract_source_urls(*chunk_groups: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered_urls: list[str] = []
    for chunk_group in chunk_groups:
        for chunk in chunk_group:
            for match in _SOURCE_URL_PATTERN.findall(chunk):
                url = _normalize_source_url(match)
                if not _is_valid_source_url(url) or url in seen:
                    continue
                seen.add(url)
                ordered_urls.append(url)
    return ordered_urls


def _extract_source_urls_from_evidence(evidence_ledger: list[EvidenceRecord]) -> list[str]:
    seen: set[str] = set()
    ordered_urls: list[str] = []
    for record in evidence_ledger:
        for url in record.source_urls:
            normalized = _normalize_source_url(url)
            if not _is_valid_source_url(normalized) or normalized in seen:
                continue
            seen.add(normalized)
            ordered_urls.append(normalized)
    return ordered_urls


def _merge_unique_urls(*url_groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in url_groups:
        for url in group:
            normalized = _normalize_source_url(url)
            if not _is_valid_source_url(normalized) or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


def _remove_internal_meta_lines(report_text: str) -> str:
    filtered_lines: list[str] = []
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped and any(pattern.search(stripped) for pattern in _INTERNAL_META_LINE_PATTERNS):
            continue
        filtered_lines.append(line)

    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sanitize_final_report_text(final_report: str) -> str:
    report_text = state_text_or_none(final_report) or FALLBACK_FINAL_REPORT
    report_text = _remove_internal_meta_lines(report_text)
    return state_text_or_none(report_text) or FALLBACK_FINAL_REPORT


def _strip_no_source_urls_sentinel(report_text: str) -> str:
    cleaned = _NO_SOURCE_URLS_SENTINEL_PATTERN.sub("", report_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _ensure_source_transparency_markers(
    final_report: str,
    note_chunks: list[str],
    raw_note_chunks: list[str],
    evidence_ledger: list[EvidenceRecord],
) -> str:
    report_text = state_text_or_none(final_report) or FALLBACK_FINAL_REPORT
    source_urls = _merge_unique_urls(
        _extract_source_urls(note_chunks, raw_note_chunks),
        _extract_source_urls_from_evidence(evidence_ledger),
        _extract_source_urls([report_text]),
    )
    report_urls = _extract_source_urls([report_text])

    report_text = _strip_no_source_urls_sentinel(report_text)
    report_text = state_text_or_none(report_text) or FALLBACK_FINAL_REPORT
    has_source_section = bool(_SOURCE_SECTION_HEADER_PATTERN.search(report_text))

    if has_source_section:
        if source_urls and not report_urls:
            source_lines = "\n".join(f"- {url}" for url in source_urls)
            return f"{report_text.rstrip()}\n{source_lines}"
        return report_text

    if source_urls:
        source_lines = "\n".join(f"- {url}" for url in source_urls)
    else:
        source_lines = "- No source URLs were available in collected notes."
    return f"{report_text.rstrip()}\n\nSources:\n{source_lines}"


def _normalized_report_inputs(state: ResearchState) -> tuple[list[str], list[str], list[EvidenceRecord]]:
    return (
        normalize_note_list(state.get("notes")),
        normalize_note_list(state.get("raw_notes")),
        normalize_evidence_ledger(state.get("evidence_ledger")),
    )


def _seed_compressed_notes(note_chunks: list[str], raw_note_chunks: list[str]) -> list[str]:
    if note_chunks or not raw_note_chunks:
        return note_chunks
    compressed_seed = compress_note_text(join_note_list(raw_note_chunks))
    if not compressed_seed:
        return note_chunks
    return [compressed_seed]


def _synthesis_chunks_for_attempt(
    note_chunks: list[str],
    raw_note_chunks: list[str],
    attempt: int,
) -> tuple[list[str], list[str]]:
    if attempt == 0:
        prompt_note_chunks = note_chunks
        prompt_raw_chunks = raw_note_chunks
    else:
        note_limit = max(1, len(note_chunks) // (attempt + 1)) if note_chunks else 0
        raw_limit = max(1, len(raw_note_chunks) // (attempt + 1)) if raw_note_chunks else 0
        prompt_note_chunks = note_chunks[:note_limit] if note_limit else []
        prompt_raw_chunks = raw_note_chunks[:raw_limit] if raw_limit else []

    # Prefer compressed notes. Raw notes are usually tool-heavy and can overflow
    # the orchestrator context window, so only include them when no compressed notes exist.
    if prompt_note_chunks:
        prompt_raw_chunks = []
    return prompt_note_chunks, prompt_raw_chunks


def _build_synthesis_payload(
    state: ResearchState,
    prompt_note_chunks: list[str],
    prompt_raw_chunks: list[str],
) -> str:
    notes_text = join_note_list(prompt_note_chunks) or "[No compressed notes]"
    raw_notes_text = join_note_list(prompt_raw_chunks) or "[No raw notes]"
    synthesis_sections = [
        f"Research brief:\n{state_text_or_none(state.get('research_brief')) or '[Missing brief]'}",
        f"Compressed notes:\n{notes_text}",
    ]
    if prompt_raw_chunks:
        synthesis_sections.append(f"Raw notes:\n{raw_notes_text}")
    synthesis_sections.append("Write the final report now.")
    return "\n\n".join(synthesis_sections)


async def _synthesize_with_retries(
    state: ResearchState,
    note_chunks: list[str],
    raw_note_chunks: list[str],
    config: RunnableConfig | None,
) -> str | None:
    model = get_llm("orchestrator")
    max_attempts = 3
    for attempt in range(max_attempts):
        prompt_note_chunks, prompt_raw_chunks = _synthesis_chunks_for_attempt(
            note_chunks,
            raw_note_chunks,
            attempt,
        )
        policy_prompt = FINAL_REPORT_PROMPT.format(current_date=today_utc_date())
        synthesis_payload = _build_synthesis_payload(state, prompt_note_chunks, prompt_raw_chunks)

        try:
            response = await invoke_runnable_with_config(
                model,
                [
                    SystemMessage(content=policy_prompt),
                    HumanMessage(content=synthesis_payload),
                ],
                config,
            )
            final_report = state_text_or_none(stringify_tool_output(response))
            if final_report:
                log_runtime_event(
                    _logger,
                    "final_report_synthesis_success",
                    attempt=attempt + 1,
                    had_compressed_notes=bool(note_chunks),
                    had_raw_notes=bool(raw_note_chunks),
                )
                return final_report
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt + 1 < max_attempts and is_token_limit_error(exc):
                log_runtime_event(
                    _logger,
                    "final_report_retry_token_limit",
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
                continue
            _logger.exception("final_report_generation synthesis failed; using fallback text")
            log_runtime_event(
                _logger,
                "final_report_fallback_exception",
                attempt=attempt + 1,
                error=str(exc),
            )
            break
    return None


def _fallback_report_text(note_chunks: list[str], raw_note_chunks: list[str]) -> str:
    compressed_fallback = join_note_list(note_chunks)
    raw_fallback = join_note_list(raw_note_chunks)
    fallback_source = "compressed_notes"
    final_report = compressed_fallback
    if not final_report:
        fallback_source = "raw_notes"
        final_report = raw_fallback
    if not final_report:
        fallback_source = "default_message"
        final_report = FALLBACK_FINAL_REPORT
    log_runtime_event(_logger, "final_report_fallback_activated", source=fallback_source)
    return final_report


def _finalize_report_text(
    final_report: str | None,
    note_chunks: list[str],
    raw_note_chunks: list[str],
    evidence_ledger: list[EvidenceRecord],
) -> str:
    report_text = final_report or _fallback_report_text(note_chunks, raw_note_chunks)
    report_text = _sanitize_final_report_text(report_text)
    return _ensure_source_transparency_markers(
        report_text,
        note_chunks,
        raw_note_chunks,
        evidence_ledger,
    )


def _final_report_update(final_report: str) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content=final_report)],
        "final_report": final_report,
        "intake_decision": "proceed",
        "awaiting_clarification": False,
    }


async def final_report_generation(
    state: ResearchState,
    config: RunnableConfig = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Create final report from supervisor outputs when not already completed."""
    note_chunks, raw_note_chunks, evidence_ledger = _normalized_report_inputs(state)
    final_report = state_text_or_none(state.get("final_report"))

    if not final_report:
        note_chunks = _seed_compressed_notes(note_chunks, raw_note_chunks)
        final_report = await _synthesize_with_retries(state, note_chunks, raw_note_chunks, config)

    final_report = _finalize_report_text(
        final_report,
        note_chunks,
        raw_note_chunks,
        evidence_ledger,
    )
    return _final_report_update(final_report)
