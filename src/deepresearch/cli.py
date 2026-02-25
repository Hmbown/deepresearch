"""Deep Research Agent CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
import webbrowser
import warnings
import uuid
from dataclasses import dataclass
from getpass import getpass
from typing import Any
from urllib.parse import urlparse
from collections.abc import Mapping

from langchain_core.messages import HumanMessage

from .config import online_evals_enabled
from .env import ensure_runtime_env_ready, project_dotenv_path, runtime_preflight, update_project_dotenv
from .message_utils import extract_text_content as _extract_text_content
from .researcher_subgraph import extract_research_from_messages
from .state import filter_evidence_ledger, normalize_evidence_ledger

_TOOL_EVENT_TYPES = {
    "on_tool_start",
    "on_tool_end",
    "on_tool_error",
    "on_chat_model_stream",
    "on_chat_model_start",
}
_PRIMARY_NODE_NAMES = {
    "scope_intake",
    "research_supervisor",
    "supervisor",
    "supervisor_prepare",
    "supervisor_finalize",
    "supervisor_terminal",
    "final_report_generation",
}

_SEARCH_SOURCE_RE = re.compile(r"^\[Source\s+(\d+)\]", re.MULTILINE)
_SEARCH_URL_RE = re.compile(r"^URL:\s*(.+)$", re.MULTILINE)
_RECURSION_LIMIT_RE = re.compile(r"Recursion limit of (\d+) reached")
_PLAN_CONFIRMATION_RE = re.compile(r"reply\s+[\"']?start[\"']?\s+.*(plan|research)", re.IGNORECASE)
_RUN_RECURSION_LIMIT = 1000


@dataclass
class _ResearchUnitContext:
    index: int
    topic: str
    depth: int
    started_at: float


def _new_thread_id() -> str:
    return uuid.uuid4().hex


def _final_assistant_text(result: dict[str, Any]) -> str:
    final_report = _extract_text_content(result.get("final_report", "")).strip()
    if final_report:
        return final_report

    messages = result.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            return _extract_text_content(getattr(message, "content", ""))
    return ""


def _result_evidence_stats(result: dict[str, Any]) -> tuple[int, int]:
    evidence = normalize_evidence_ledger(result.get("evidence_ledger"))
    fetched_evidence = filter_evidence_ledger(evidence, source_type="fetched")
    domains = _collect_domains(fetched_evidence)
    return len(fetched_evidence), len(domains)


def _result_section_title(result: dict[str, Any], elapsed_seconds: float | None = None) -> str:
    if result.get("intake_decision") == "clarify":
        latest_text = _final_assistant_text(result)
        if _PLAN_CONFIRMATION_RE.search(latest_text):
            if elapsed_seconds is None:
                return "RESEARCH PLAN"
            return f"RESEARCH PLAN ({_format_duration(elapsed_seconds)})"
        if elapsed_seconds is None:
            return "CLARIFICATION"
        return f"CLARIFICATION ({_format_duration(elapsed_seconds)})"

    evidence_count, domain_count = _result_evidence_stats(result)
    parts = [f"{evidence_count} sources", f"{domain_count} domains"]
    if elapsed_seconds is not None:
        parts.append(_format_duration(elapsed_seconds))
    return f"RESEARCH REPORT ({' | '.join(parts)})"


def _format_duration(elapsed: float) -> str:
    if elapsed < 60:
        return f"{elapsed:.0f}s"

    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes}m {seconds:02d}s"


def _format_section_header(name: str, width: int = 66) -> str:
    name = name.strip()
    if not name:
        return "-" * width
    if width <= len(name) + 4:
        return name
    return f"-- {name} " + "-" * (width - len(name) - 4)


def _format_phase_header(name: str, width: int = 70) -> str:
    label = f"=== {name.strip()} "
    if len(label) >= width:
        return label.strip()
    return label + "=" * (width - len(label))


def _clean_tool_failure_detail(detail: str) -> str:
    cleaned = detail.strip()
    if cleaned.startswith("[Research unit failed:"):
        cleaned = cleaned[len("[Research unit failed:") :].strip()
    if cleaned.endswith("]"):
        cleaned = cleaned[:-1].strip()
    cleaned = re.sub(r"For troubleshooting, visit:.*", "", cleaned, flags=re.DOTALL).strip()
    return cleaned


def _summarize_research_failure(detail: str, unit_label: str) -> str:
    cleaned = _clean_tool_failure_detail(detail)
    recursion_match = _RECURSION_LIMIT_RE.search(cleaned)
    if recursion_match:
        step_limit = recursion_match.group(1)
        return f"{unit_label} encountered a recursion limit ({step_limit} steps); continuing with other research units."
    return f"{unit_label} ended early: {_truncate(cleaned, 180)}"


def _extract_query_arg(value: Any, key: str) -> str:
    if isinstance(value, Mapping):
        arg = value.get(key)
        if isinstance(arg, str):
            return arg.strip()
    return ""


def _truncate(text: str, max_chars: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 0:
        return ""
    return normalized[: max_chars - 1] + "..."


def _parse_search_summary(raw: str) -> tuple[int, list[str]]:
    if not isinstance(raw, str):
        return 0, []

    if "No relevant search results found" in raw:
        return 0, []

    source_count = len(_SEARCH_SOURCE_RE.findall(raw))

    domains: list[str] = []
    for match in _SEARCH_URL_RE.finditer(raw):
        url = match.group(1).strip()
        if not url or url == "N/A":
            continue
        domain = urlparse(url).netloc.lower()
        if domain and domain not in domains:
            domains.append(domain)

    return source_count, domains


def _format_domain_list(domains: list[str], max_items: int = 3) -> str:
    if not domains:
        return ""
    if len(domains) <= max_items:
        return ", ".join(domains)
    return f"{', '.join(domains[:max_items])}, ..."


def _extract_supervisor_progress_payload(event_name: str, data: Mapping[str, Any] | None) -> dict[str, Any]:
    if event_name != "supervisor_progress" or not isinstance(data, Mapping):
        return {}
    return dict(data)


def _collect_evidence_from_research_output(output: Mapping[str, Any] | None) -> tuple[list[Any], int]:
    if not isinstance(output, Mapping):
        return [], 0

    evidence_records = normalize_evidence_ledger(output.get("evidence_ledger"))
    if not evidence_records and isinstance(output.get("messages"), list):
        _, _, extracted_evidence = extract_research_from_messages({"messages": list(output["messages"])})
        evidence_records = extracted_evidence

    fetched_records = filter_evidence_ledger(evidence_records, source_type="fetched")
    model_cited_records = filter_evidence_ledger(evidence_records, source_type="model_cited")
    return fetched_records, len(model_cited_records)


def _collect_domains(records: list[Any]) -> set[str]:
    domains: set[str] = set()
    for record in records:
        for raw_url in getattr(record, "source_urls", []):
            domain = urlparse(str(raw_url)).netloc.lower()
            if domain:
                domains.add(domain)
    return domains


def _extract_topic_from_research_chain_input(data: Mapping[str, Any] | None) -> str:
    if not isinstance(data, Mapping):
        return ""
    payload = data.get("input")
    if not isinstance(payload, Mapping):
        return ""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if isinstance(message, Mapping):
            message_type = str(message.get("type", "")).lower()
            if message_type not in {"human", "user"}:
                continue
            content = _extract_text_content(message.get("content", "")).strip()
            if content:
                return content
            continue

        if getattr(message, "type", "") != "human":
            continue
        content = _extract_text_content(getattr(message, "content", "")).strip()
        if content:
            return content
    return ""


class ProgressDisplay:
    """Small progress printer for run-time events."""

    def __init__(self, *, verbose: bool = False, quiet: bool = False, stream: Any = sys.stderr) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self.stream = stream

        self._start = time.monotonic()
        self._next_research_unit = 1
        self._active_research: dict[str, _ResearchUnitContext] = {}

        self._evidence_keys: set[str] = set()
        self._evidence_record_count = 0
        self._evidence_domains: set[str] = set()
        self._model_cited_record_count = 0

        self._current_phase: str | None = None
        self._phase_started_at: dict[str, float] = {}
        self._research_wave = 0
        self._active_wave = 0
        self._wave_started_at: dict[int, float] = {}
        self._wave_plan_announced: set[int] = set()
        self._planned_track_index_by_call_id: dict[str, int] = {}
        self._pipeline_finished = False
        self._last_top_level_output: dict[str, Any] = {}

    def _depth(self, metadata: Mapping[str, Any] | None) -> int:
        if not isinstance(metadata, Mapping):
            return 0
        checkpoint_ns = str(metadata.get("checkpoint_ns", ""))
        if not checkpoint_ns:
            return 0
        return checkpoint_ns.count("|")

    def _emit(self, text: str, depth: int = 0) -> None:
        if self.quiet:
            return
        indent = "  " * depth
        print(f"{indent}{text}", file=self.stream, flush=True)

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def latest_output(self) -> dict[str, Any]:
        return dict(self._last_top_level_output)

    def finish_if_needed(self) -> None:
        self._finish_pipeline()

    def _is_top_level_graph_event(self, metadata: Mapping[str, Any] | None) -> bool:
        if not isinstance(metadata, Mapping):
            return True
        checkpoint_ns = str(metadata.get("checkpoint_ns", "")).strip()
        return not checkpoint_ns

    def _start_phase(self, phase_name: str) -> None:
        now = time.monotonic()
        if self._current_phase == phase_name:
            return

        if self._current_phase and self._current_phase in self._phase_started_at:
            elapsed = now - self._phase_started_at[self._current_phase]
            self._emit(f"{self._current_phase} complete in {_format_duration(elapsed)}", depth=0)

        self._current_phase = phase_name
        self._phase_started_at[phase_name] = now
        self._emit(_format_phase_header(phase_name), depth=0)

    def _finish_pipeline(self) -> None:
        if self._pipeline_finished:
            return

        now = time.monotonic()
        if self._current_phase and self._current_phase in self._phase_started_at:
            phase_elapsed = now - self._phase_started_at[self._current_phase]
            self._emit(f"{self._current_phase} complete in {_format_duration(phase_elapsed)}", depth=0)

        self._emit(f"Full pipeline finished in {_format_duration(self.elapsed())}", depth=0)
        self._pipeline_finished = True

    def _active_research_context(self, checkpoint_ns: str | None) -> _ResearchUnitContext | None:
        if not checkpoint_ns:
            return None
        matches = [
            (section_ns, context)
            for section_ns, context in self._active_research.items()
            if checkpoint_ns.startswith(section_ns)
        ]
        if not matches:
            return None
        # Prefer the deepest active section.
        return sorted(matches, key=lambda item: len(item[0]), reverse=True)[0][1]

    def _tool_depth(self, checkpoint_ns: str | None) -> int:
        context = self._active_research_context(checkpoint_ns)
        if context is not None:
            return context.depth + 1
        base_depth = self._depth({"checkpoint_ns": checkpoint_ns or ""} if checkpoint_ns is not None else None)
        return max(0, base_depth + 1)

    def _add_evidence(self, raw_records: Any) -> None:
        records = normalize_evidence_ledger(raw_records)
        for record in records:
            source_urls = [str(url).strip() for url in getattr(record, "source_urls", []) if str(url).strip()]
            key = ",".join(sorted(set(source_urls)))
            if not key or key in self._evidence_keys:
                continue
            self._evidence_keys.add(key)
            self._evidence_record_count += 1
            self._evidence_domains.update(_collect_domains([record]))

    def _start_research_section(
        self,
        checkpoint_ns: str,
        metadata: Mapping[str, Any] | None,
        data: Mapping[str, Any] | None,
    ) -> None:
        if checkpoint_ns in self._active_research:
            return

        topic_from_input = _extract_topic_from_research_chain_input(data)
        if topic_from_input:
            topic = topic_from_input
        else:
            topic = "[topic unavailable]"
        context = _ResearchUnitContext(
            index=self._next_research_unit,
            topic=topic,
            depth=max(1, self._depth(metadata)),
            started_at=time.monotonic(),
        )
        self._next_research_unit += 1
        self._active_research[checkpoint_ns] = context

        if self.verbose:
            self._emit(_format_section_header(f"researcher[{context.index}]", width=58), depth=context.depth)
            self._emit(f"Topic: {topic}", depth=context.depth + 1)
        else:
            self._emit(f'researcher[{context.index}] "{_truncate(topic, 84)}" started', depth=1)

    def _finish_research_section(self, checkpoint_ns: str, output: Mapping[str, Any]) -> None:
        context = self._active_research.pop(checkpoint_ns, None)
        if context is None:
            return

        evidence_records, model_cited_count = _collect_evidence_from_research_output(output)
        self._add_evidence(evidence_records)
        self._model_cited_record_count += max(0, model_cited_count)
        domains = _collect_domains(evidence_records)
        elapsed = _format_duration(time.monotonic() - context.started_at)
        summary = (
            f'researcher[{context.index}] "{_truncate(context.topic, 72)}" - '
            f"{len(evidence_records)} sources, {len(domains)} domains ({elapsed})"
        )
        if model_cited_count > 0:
            summary = (
                f'researcher[{context.index}] "{_truncate(context.topic, 72)}" - '
                f"{len(evidence_records)} sources, {len(domains)} domains, "
                f"{model_cited_count} model-cited URLs ({elapsed})"
            )
        self._emit(
            summary,
            depth=1,
        )
        self._emit(
            (
                f"Total so far: {self._evidence_record_count} sources "
                f"from {len(self._evidence_domains)} domains"
            ),
            depth=2,
        )

    def _coerce_non_negative_int(self, value: Any, default: int = 0) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, parsed)

    def _sync_runtime_evidence_totals(self, progress: Mapping[str, Any]) -> None:
        evidence_count = self._coerce_non_negative_int(progress.get("evidence_record_count"), self._evidence_record_count)
        source_domains = progress.get("source_domains")
        if isinstance(source_domains, list):
            normalized_domains = {str(domain).strip().lower() for domain in source_domains if str(domain).strip()}
            self._evidence_domains = normalized_domains
        self._evidence_record_count = evidence_count
        self._model_cited_record_count = self._coerce_non_negative_int(
            progress.get("model_cited_record_count"),
            self._model_cited_record_count,
        )

    def _handle_research_unit_summary(self, summary: Mapping[str, Any], depth: int) -> None:
        status = str(summary.get("status", "")).strip().lower()
        topic = _truncate(str(summary.get("topic") or "[topic unavailable]"), 72)
        call_id = str(summary.get("call_id") or "").strip()
        planned_track_index = self._planned_track_index_by_call_id.get(call_id, 0) if call_id else 0
        track_prefix = f"track[{planned_track_index}] " if planned_track_index > 0 else ""
        try:
            duration_seconds = max(0.0, float(summary.get("duration_seconds", 0.0)))
        except (TypeError, ValueError):
            duration_seconds = 0.0
        duration_label = _format_duration(duration_seconds)

        if status == "failed":
            reason = str(summary.get("failure_reason") or "research unit execution failed")
            unit_label = f'{track_prefix}"{topic}"'.strip()
            self._emit(f"{_summarize_research_failure(reason, unit_label)} ({duration_label})", depth=max(1, depth))
        elif status == "empty":
            self._emit(f'{track_prefix}"{topic}" returned no usable notes', depth=max(1, depth))
        elif status == "skipped":
            self._emit(f'{track_prefix}"{topic}" deferred due to runtime caps.', depth=max(1, depth))
        elif status == "missing_topic":
            if planned_track_index > 0:
                self._emit(
                    f"{track_prefix}Supervisor issued ConductResearch without a research_topic.",
                    depth=max(1, depth),
                )
            else:
                self._emit("Supervisor issued ConductResearch without a research_topic.", depth=max(1, depth))

    def _render_planned_research_tracks(self, progress: Mapping[str, Any], wave_index: int) -> None:
        if wave_index in self._wave_plan_announced:
            return

        planned_units = progress.get("planned_research_units")
        if not isinstance(planned_units, list):
            return

        track_entries: list[tuple[str, str]] = []
        for item in planned_units:
            if not isinstance(item, Mapping):
                continue
            topic = str(item.get("topic") or "").strip()
            if not topic:
                continue
            call_id = str(item.get("call_id") or "").strip()
            track_entries.append((call_id, topic))
        if not track_entries:
            return

        self._emit("Research tracks:", depth=1)
        for index, (_, topic) in enumerate(track_entries, start=1):
            self._emit(f"[{index}] {_truncate(topic, 92)}", depth=2)
        for index, (call_id, _) in enumerate(track_entries, start=1):
            if not call_id:
                continue
            self._planned_track_index_by_call_id[call_id] = index
        self._wave_plan_announced.add(wave_index)

    def _handle_supervisor_runtime_progress(self, progress: Mapping[str, Any], depth: int) -> None:
        self._sync_runtime_evidence_totals(progress)
        requested_count = self._coerce_non_negative_int(progress.get("requested_research_units"), 0)
        dispatch_count = self._coerce_non_negative_int(progress.get("dispatched_research_units"), 0)
        skipped_count = self._coerce_non_negative_int(progress.get("skipped_research_units"), 0)
        supervisor_iteration = self._coerce_non_negative_int(progress.get("supervisor_iteration"), 1)

        if dispatch_count > 0:
            wave_index = self._active_wave
            if wave_index == 0:
                self._research_wave += 1
                wave_index = self._research_wave
                self._active_wave = wave_index
                self._wave_started_at[wave_index] = time.monotonic()
            message = (
                f"Wave {wave_index}: Supervisor iteration {supervisor_iteration} - "
                f"dispatching {dispatch_count} researcher{'s' if dispatch_count != 1 else ''} in parallel"
            )
            deferred_count = max(skipped_count, max(0, requested_count - dispatch_count))
            if deferred_count > 0:
                message += f" ({deferred_count} deferred by runtime caps)"
            self._emit(message, depth=0)
            self._render_planned_research_tracks(progress, wave_index)
        else:
            self._emit(f"Supervisor iteration {supervisor_iteration}: evaluating quality gate.", depth=0)

        research_units = progress.get("research_units")
        if isinstance(research_units, list):
            for summary in research_units:
                if isinstance(summary, Mapping):
                    self._handle_research_unit_summary(summary, depth)

        if dispatch_count > 0:
            wave_index = self._active_wave
            wave_elapsed = _format_duration(time.monotonic() - self._wave_started_at.get(wave_index, self._start))
            self._emit(
                (
                    f"Wave {wave_index} complete in {wave_elapsed}: "
                    f"{self._evidence_record_count} sources, {len(self._evidence_domains)} domains"
                ),
                depth=0,
            )
        self._active_wave = 0

    def _handle_chain_start(
        self, event_name: str, metadata: Mapping[str, Any] | None, data: Mapping[str, Any] | None
    ) -> None:
        if self.verbose and event_name in _PRIMARY_NODE_NAMES:
            self._emit(_format_section_header(event_name), depth=self._depth(metadata))

        if event_name == "scope_intake":
            self._start_phase("PHASE 1: INTAKE")
            return

        if event_name == "research_supervisor":
            self._start_phase("PHASE 2: RESEARCH")
            return

        if event_name == "supervisor_prepare":
            return

        if event_name == "final_report_generation":
            self._start_phase("PHASE 3: SYNTHESIS")
            self._emit(
                (
                    f"Synthesizing from {self._evidence_record_count} sources "
                    f"and {len(self._evidence_domains)} domains..."
                ),
                depth=1,
            )
            return

        checkpoint_ns = str((metadata or {}).get("checkpoint_ns", "")) if isinstance(metadata, Mapping) else ""
        if checkpoint_ns and checkpoint_ns.count("|") == 1:
            self._start_research_section(checkpoint_ns, metadata, data)

    def _handle_chain_end(self, event_name: str, metadata: Mapping[str, Any] | None, data: Mapping[str, Any] | None) -> None:
        if not isinstance(data, Mapping):
            return

        checkpoint_ns = str((metadata or {}).get("checkpoint_ns", "")) if isinstance(metadata, Mapping) else ""
        if checkpoint_ns in self._active_research and event_name in {"LangGraph", "deep-researcher"}:
            output = data.get("output")
            if isinstance(output, Mapping):
                self._finish_research_section(checkpoint_ns, output)

    def _handle_chain_error(
        self, metadata: Mapping[str, Any] | None, data: Mapping[str, Any] | None
    ) -> None:
        checkpoint_ns = str((metadata or {}).get("checkpoint_ns", "")) if isinstance(metadata, Mapping) else ""
        if checkpoint_ns not in self._active_research:
            return

        context = self._active_research.pop(checkpoint_ns, None)
        if context is None:
            return

        error_text = _extract_text_content((data or {}).get("error", "")) if isinstance(data, Mapping) else ""
        failure_detail = error_text or "[Research unit failed]"
        elapsed = _format_duration(time.monotonic() - context.started_at)
        label = f'researcher[{context.index}] "{_truncate(context.topic, 72)}"'
        self._emit(f"{_summarize_research_failure(failure_detail, label)} ({elapsed})", depth=1)

    def _handle_tool_event(
        self,
        event_type: str,
        event_name: str,
        metadata: Mapping[str, Any] | None,
        data: Mapping[str, Any] | None,
    ) -> None:
        if not isinstance(data, Mapping):
            return

        checkpoint_ns = str((metadata or {}).get("checkpoint_ns", "")) if isinstance(metadata, Mapping) else ""
        depth = self._tool_depth(checkpoint_ns)
        if event_type == "on_tool_start":
            if not self.verbose and event_name in {"search_web", "fetch_url", "think_tool"}:
                return
            if event_name == "search_web":
                query = _extract_query_arg(data.get("input"), "query")
                self._emit(f'[search] "{query}"', depth=depth)
            elif event_name == "fetch_url":
                fetch_url = _extract_query_arg(data.get("input"), "url")
                self._emit(f"[fetch] {fetch_url}", depth=depth)
            elif event_name == "think_tool":
                self._emit("[think]", depth=depth)
            elif self.verbose:
                self._emit(f"[tool start: {event_name}]", depth=depth)
            return

        if event_type == "on_tool_end":
            if event_name == "search_web":
                output = _extract_text_content(data.get("output", ""))
                if not self.verbose:
                    return
                source_count, domains = _parse_search_summary(output)
                domain_label = _format_domain_list(domains)
                if domain_label:
                    self._emit(f"-> {source_count} results ({domain_label})", depth=depth)
                else:
                    self._emit(f"-> {source_count} results", depth=depth)
                if self.verbose and output:
                    self._emit(f"[search] {_truncate(output, 180)}", depth=depth + 1)
                return

            if event_name == "fetch_url":
                output = _extract_text_content(data.get("output", ""))
                if output.startswith("[Fetch failed"):
                    self._emit(f"[fail] {output}", depth=depth)
                elif self.verbose:
                    self._emit("[fetch] done", depth=depth)
                return

            if event_name == "think_tool":
                if not self.verbose:
                    return
                output = _extract_text_content(data.get("output", ""))
                if output.startswith("Reflection recorded:"):
                    output = output.split(":", 1)[1].strip()
                self._emit(f"[think] {_truncate(output, 120)}", depth=depth)
                return

            if self.verbose:
                self._emit(f"[tool end: {event_name}]", depth=depth)
            return

        if event_type == "on_tool_error":
            error = _extract_text_content(data.get("error", ""))
            self._emit(f"Tool issue in {event_name}: {_truncate(error, 180)}", depth=depth)
            return

        if event_type == "on_chat_model_stream" and self.verbose:
            chunk = data.get("chunk")
            text = _extract_text_content(getattr(chunk, "content", chunk))
            if text:
                self._emit(f"[model] {_truncate(text, 120)}", depth=depth)

    def _handle_custom_event(self, event_name: str, metadata: Mapping[str, Any] | None, data: Mapping[str, Any] | None) -> None:
        progress = _extract_supervisor_progress_payload(event_name, data)
        if not progress:
            return
        self._handle_supervisor_runtime_progress(progress, self._depth(metadata) + 1)

    def handle_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        event_type = str(event.get("event", ""))
        name = str(event.get("name", ""))
        metadata = event.get("metadata") if isinstance(event.get("metadata"), Mapping) else None
        data = event.get("data") if isinstance(event.get("data"), Mapping) else {}

        if event_type == "on_chain_start":
            self._handle_chain_start(name, metadata, data)

        if event_type in _TOOL_EVENT_TYPES:
            self._handle_tool_event(event_type, name, metadata, data)

        if event_type == "on_custom_event":
            self._handle_custom_event(name, metadata, data)

        if event_type == "on_chain_error":
            self._handle_chain_error(metadata, data)

        if event_type == "on_chain_end":
            self._handle_chain_end(name, metadata, data)
            output = event.get("data", {}).get("output", {}) if isinstance(event.get("data"), Mapping) else {}
            if self._is_top_level_graph_event(metadata) and isinstance(output, Mapping):
                self._last_top_level_output = dict(output)
                if name == "LangGraph" or "intake_decision" in output or "final_report" in output:
                    self._finish_pipeline()
                    return dict(output)

        return None


def _get_app():
    from .graph import app

    return app


def _thread_config(thread_id: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _RUN_RECURSION_LIMIT,
    }
    if online_evals_enabled():
        from .evals import attach_online_eval_callback

        cfg = attach_online_eval_callback(cfg)
    return cfg


async def _run_with_progress(
    app: Any,
    payload: dict[str, Any],
    config: dict[str, Any],
    *,
    verbose: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    if quiet:
        return await app.ainvoke(payload, config=config)

    display = ProgressDisplay(verbose=verbose, quiet=quiet)
    final_output: dict[str, Any] = {}

    async for event in app.astream_events(payload, config=config, version="v2"):
        output = display.handle_event(event)
        if isinstance(output, Mapping):
            final_output = dict(output)

    if not final_output:
        final_output = display.latest_output()
        if final_output:
            display.finish_if_needed()

    return final_output


async def run(
    query: str,
    thread_id: str | None = None,
    prior_messages: list[Any] | None = None,
    *,
    quiet: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a deep research query and return the agent result state."""
    ensure_runtime_env_ready()

    resolved_thread_id = (thread_id or "").strip() or _new_thread_id()
    payload_messages = list(prior_messages or [])
    payload_messages.append(HumanMessage(content=query))
    app = _get_app()
    payload = {"messages": payload_messages}
    config = _thread_config(resolved_thread_id)

    if not hasattr(app, "astream_events"):
        return await app.ainvoke(payload, config=config)

    return await _run_with_progress(app, payload, config, verbose=verbose, quiet=quiet)


def print_results(result: dict[str, Any], elapsed_seconds: float | None = None) -> None:
    """Print the final assistant response."""
    response_text = _final_assistant_text(result)
    if response_text:
        section_title = _result_section_title(result, elapsed_seconds=elapsed_seconds)
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
    if not ok:
        dotenv_path = project_dotenv_path()
        template_path = dotenv_path.with_name(".env.example")
        print("\nQuick setup")
        print("-" * 40)
        if not dotenv_path.exists():
            if template_path.exists():
                print(f"cp {template_path} {dotenv_path}")
            else:
                print(f"touch {dotenv_path}")
        print(f"echo 'OPENAI_API_KEY=YOUR_OPENAI_API_KEY' >> {dotenv_path}")
        print(f"echo 'SEARCH_PROVIDER=none' >> {dotenv_path}  # optional quick start")
    return 0 if ok else 1


def _prompt_required_value(prompt: str, *, secret: bool = False) -> str:
    while True:
        value = (getpass(prompt) if secret else input(prompt)).strip()
        if value:
            return value
        print("This field is required.")


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter yes or no.")


def _prompt_search_provider() -> str:
    supported = {"exa", "tavily", "none"}
    while True:
        raw = input("Search provider [exa/tavily/none] (default: exa): ").strip().lower()
        if not raw:
            return "exa"
        if raw in supported:
            return raw
        print("Choose one of: exa, tavily, none.")


def _print_setup_header() -> None:
    print("Deep Research Setup")
    print("-" * 40)
    print("Required fields are marked in prompts.")


def _collect_setup_inputs() -> tuple[dict[str, str], bool, str | None]:
    openai_key = _prompt_required_value("OpenAI API key (required): ", secret=True)

    search_provider = _prompt_search_provider()
    updates: dict[str, str] = {
        "OPENAI_API_KEY": openai_key,
        "SEARCH_PROVIDER": search_provider,
    }

    if search_provider == "exa":
        updates["EXA_API_KEY"] = _prompt_required_value("Exa API key (required for provider=exa): ", secret=True)
    elif search_provider == "tavily":
        updates["TAVILY_API_KEY"] = _prompt_required_value(
            "Tavily API key (required for provider=tavily): ",
            secret=True,
        )

    langsmith_enabled = _prompt_yes_no("Enable LangSmith tracing?", default=False)
    langsmith_project: str | None = None
    if langsmith_enabled:
        updates["LANGCHAIN_TRACING_V2"] = "true"
        updates["LANGCHAIN_API_KEY"] = _prompt_required_value(
            "LangSmith API key (required when tracing is enabled): ",
            secret=True,
        )
        langsmith_project = input("LangSmith project (optional, default: deepresearch-local): ").strip()
        if not langsmith_project:
            langsmith_project = "deepresearch-local"
        updates["LANGCHAIN_PROJECT"] = langsmith_project
    else:
        updates["LANGCHAIN_TRACING_V2"] = "false"

    return updates, langsmith_enabled, langsmith_project


def _print_setup_summary(
    dotenv_path: str,
    updates: Mapping[str, str],
    *,
    langsmith_enabled: bool,
    langsmith_project: str | None,
) -> None:
    search_provider = updates.get("SEARCH_PROVIDER", "exa")
    print(f"\nWrote setup to `{dotenv_path}`")
    print("\nConfigured services")
    print("-" * 40)
    print("OpenAI API key: set")
    print(f"Search provider: {search_provider}")
    if search_provider == "exa":
        print("EXA_API_KEY: set")
    elif search_provider == "tavily":
        print("TAVILY_API_KEY: set")
    else:
        print("Search API key: not required")

    if langsmith_enabled:
        print("LangSmith: enabled")
        print("LANGCHAIN_API_KEY: set")
        print(f"LANGCHAIN_PROJECT: {langsmith_project}")
    else:
        print("LangSmith: disabled")


def _maybe_open_langsmith(langsmith_enabled: bool) -> None:
    if not langsmith_enabled:
        return
    if not _prompt_yes_no("Open LangSmith in browser now?", default=False):
        return
    try:
        webbrowser.open("https://smith.langchain.com/")
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}")


def _run_post_setup_preflight(langsmith_enabled: bool, langsmith_project: str | None) -> int:
    print("\nRunning preflight...")
    result_code = print_preflight(project_name=langsmith_project if langsmith_enabled else None)
    if result_code == 0:
        print("\nSetup complete, ready to run: deepresearch 'your query'")
    return result_code


def run_setup_wizard() -> int:
    """Run interactive setup and verify configuration with preflight."""
    _print_setup_header()
    updates, langsmith_enabled, langsmith_project = _collect_setup_inputs()
    dotenv_path = update_project_dotenv(updates)
    _print_setup_summary(
        str(dotenv_path),
        updates,
        langsmith_enabled=langsmith_enabled,
        langsmith_project=langsmith_project,
    )
    _maybe_open_langsmith(langsmith_enabled)
    return _run_post_setup_preflight(langsmith_enabled, langsmith_project)


async def run_session(thread_id: str, *, quiet: bool = False, verbose: bool = False) -> None:
    """Run an interactive multi-turn session on a single thread."""
    from langgraph.checkpoint.memory import MemorySaver
    from .graph import build_app

    ensure_runtime_env_ready()
    app = build_app(checkpointer=MemorySaver())
    config = _thread_config(thread_id)

    print(f"Session thread_id: {thread_id}")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q", "/exit"}:
            return

        payload = {"messages": [HumanMessage(content=query)]}
        result = await _run_with_progress(app, payload, config, verbose=verbose, quiet=quiet)
        print_results(result)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepresearch",
        description="Deep Research Agent CLI",
    )
    parser.add_argument(
        "--preflight",
        nargs="?",
        const="",
        metavar="PROJECT",
        help="Run runtime preflight checks and exit. Optionally pass a project name.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run interactive setup wizard and exit.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed tool-level runtime events.",
    )
    verbosity.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress streaming progress output.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Research query. Omit to start an interactive session.",
    )
    return parser


def main() -> None:
    # Suppress noisy Pydantic serialization warnings from langchain-openai Responses API.
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

    parser = _build_arg_parser()
    parsed = parser.parse_args(sys.argv[1:])

    if parsed.setup and parsed.preflight is not None:
        parser.error("--setup cannot be combined with --preflight.")

    if parsed.setup and parsed.query:
        parser.error("--setup does not accept a research query.")

    if parsed.setup:
        raise SystemExit(run_setup_wizard())

    if parsed.preflight is not None:
        project_name = str(parsed.preflight).strip() or None
        raise SystemExit(print_preflight(project_name=project_name))

    query = " ".join(parsed.query).strip()
    if query:
        thread_id = _new_thread_id()
        print(f"\nResearching: {query}\n")
        started = time.monotonic()
        result = asyncio.run(run(query, thread_id=thread_id, quiet=bool(parsed.quiet), verbose=bool(parsed.verbose)))
        print_results(result, elapsed_seconds=time.monotonic() - started)
        return

    print("Deep Research Agent")
    print("-" * 40)
    provided_thread = input("Thread ID (optional, press enter to auto-generate): ").strip()
    thread_id = provided_thread or _new_thread_id()
    asyncio.run(run_session(thread_id, quiet=bool(parsed.quiet), verbose=bool(parsed.verbose)))


if __name__ == "__main__":
    main()
