"""Shared state models and runtime helpers for the native LangGraph path."""

from __future__ import annotations

import json
import operator
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ValidationError

from .config import get_supervisor_notes_max_bullets, get_supervisor_notes_word_budget
from .message_utils import extract_text_content

FALLBACK_CLARIFY_QUESTION = (
    "Before I start, can you clarify the scope so I can research the right thing?"
)
FALLBACK_VERIFICATION = (
    "I have enough context to begin deep research now. I will break this into focused research "
    "tracks, verify with strong sources, and return a synthesized report with citations."
)
FALLBACK_FINAL_REPORT = (
    "I completed research execution but did not receive a final synthesis payload. "
    "Please ask a focused follow-up and I will refine from the collected notes."
)
FALLBACK_SUPERVISOR_NO_USEFUL_RESEARCH = (
    "I could not produce usable research findings from the current scope in this run. "
    "Please narrow the request (topic angle, timeframe, or source requirements) and I will retry."
)
FALLBACK_FOLLOWUP_CLARIFICATION = (
    "I want to ensure we are aligned: is your latest question still about the same research topic?"
)
FALLBACK_SCOPE_BOUNDARY_CLARIFICATION = (
    "Before I start, can you set the scope for this broad request: what timeframe should I cover, "
    "and what geography or target universe should I include?"
)
FALLBACK_PLAN_CONFIRMATION_FOOTER = (
    'If this plan looks right, reply "start". If you want changes, tell me what to adjust.'
)
FOLLOW_UP_CONTINUATION_MARKERS = {
    "also",
    "additionally",
    "add",
    "further",
    "more",
    "detail",
    "details",
    "specific",
    "specifically",
    "focus",
    "refine",
    "refinement",
    "deeper",
    "dive",
    "expand",
}
FOLLOW_UP_SHIFT_MARKERS = {
    "instead",
    "change",
    "switch",
    "different",
    "new",
    "another",
    "unrelated",
    "otherwise",
    "topic",
    "let",
    "lets",
    "let's",
}
STRONG_FOLLOW_UP_SHIFT_MARKERS = {
    "instead",
    "change",
    "switch",
    "different",
    "another",
    "unrelated",
}
STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "to",
    "with",
    "within",
    "about",
    "can",
    "how",
    "i",
    "i'm",
    "im",
    "you",
    "we",
    "they",
    "them",
    "me",
    "us",
}
_BROAD_SCOPE_NOUN_MARKERS = {
    "benchmark",
    "benchmarks",
    "companies",
    "countries",
    "framework",
    "frameworks",
    "industry",
    "industries",
    "laws",
    "markets",
    "models",
    "platforms",
    "policies",
    "providers",
    "regions",
    "risks",
    "sectors",
    "startups",
    "stocks",
    "tools",
    "trends",
    "universities",
    "vendors",
    "watchlist",
}
_BROAD_SCOPE_ACTION_MARKERS = {
    "best",
    "compare",
    "find",
    "identify",
    "least",
    "list",
    "most",
    "rank",
    "screen",
    "top",
    "what are",
    "which",
}
_SCOPE_TIMEFRAME_MARKERS = {
    "after",
    "as of",
    "before",
    "between",
    "from",
    "in 20",
    "latest",
    "next",
    "onward",
    "recent",
    "since",
    "through",
    "throughout",
    "today",
    "until",
    "yesterday",
}
_SCOPE_GEOGRAPHY_MARKERS = {
    "africa",
    "america",
    "asia",
    "australia",
    "canada",
    "china",
    "europe",
    "global",
    "india",
    "japan",
    "latin america",
    "middle east",
    "uk",
    "u.k.",
    "u.s.",
    "united kingdom",
    "united states",
    "usa",
    "worldwide",
}
_SCOPE_UNIVERSE_MARKERS = {
    "exchange",
    "large cap",
    "listed",
    "microcap",
    "mid cap",
    "nasdaq",
    "nyse",
    "otc",
    "public company",
    "russell",
    "s&p",
    "small cap",
}
_SCOPE_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_SCOPE_QUARTER_PATTERN = re.compile(r"\bq[1-4]\b")
_SCOPE_MONTH_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
)


class ClarifyWithUser(BaseModel):
    """Structured decision for whether a clarification turn is needed."""

    need_clarification: bool = Field(
        description="Whether the user should be asked clarification questions before research.",
    )
    question: str = Field(
        description="Up to 3 conversational scope-narrowing questions bundled in a single message. Ask everything you need in one turn so the user can answer all at once.",
    )
    verification: str = Field(
        description="Acknowledgement message that research is starting when clarification is not required.",
    )


class ResearchBrief(BaseModel):
    """Structured research brief used to seed the supervisor."""

    research_brief: str = Field(
        description="Focused, concrete research brief synthesized from the full chat history.",
    )


class ResearchPlan(BaseModel):
    """Short research plan preview for user approval. Keep every field concise."""

    scope: str = Field(description="One sentence: what will be researched and any boundaries.")
    research_tracks: list[str] = Field(
        description="Research tracks to investigate. Each track is ONE sentence describing the angle, not the methodology. Merge related angles rather than splitting. Do not include scope-clarification tracks.",
    )
    evidence_strategy: str = Field(
        description="One sentence: what source types to prioritize.",
    )
    output_format: str = Field(
        description="One sentence: deliverable format (e.g. 'memo-style report with inline citations').",
    )


class ConductResearch(BaseModel):
    """Tool payload for delegating one focused research unit."""

    research_topic: str = Field(
        description=(
            "The focused topic for one research unit with enough context to execute independently."
        ),
    )


class ResearchComplete(BaseModel):
    """Tool payload signaling that supervisor synthesis is complete."""

    pass


class EvidenceRecord(BaseModel):
    """Structured evidence item extracted from researcher output."""

    claim: str = Field(description="Atomic factual claim or finding text.")
    source_urls: list[str] = Field(
        default_factory=list,
        description="Source URLs that support this claim.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Deterministic confidence estimate in [0, 1].",
    )
    contradiction_or_uncertainty: str | None = Field(
        default=None,
        description="Contradiction or uncertainty note, if present.",
    )


class SupervisorState(TypedDict, total=False):
    """Supervisor state that coordinates multiple researcher delegations."""

    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    notes: Annotated[list[str], operator.add]
    research_iterations: int
    raw_notes: Annotated[list[str], operator.add]
    evidence_ledger: Annotated[list[EvidenceRecord], operator.add]
    pending_research_calls: list[dict[str, Any]]
    pending_complete_calls: list[dict[str, Any]]
    pending_requested_research_units: int
    pending_dispatched_research_units: int
    pending_skipped_research_units: int
    pending_remaining_iterations: int
    research_unit_summaries: Annotated[list[dict[str, Any]], operator.add]
    research_unit_summaries_consumed: int
    supervisor_exception: str | None


class ResearchState(MessagesState):
    """Main graph state for intake + supervisor + final synthesis."""

    research_brief: str | None = None
    intake_decision: Literal["clarify", "proceed"] | None = None
    awaiting_clarification: bool = False
    # These fields are overwritten at intake handoff and supervisor completion.
    # Using additive reducers here can silently retain stale prior-turn state under
    # checkpointed/persisted threads.
    supervisor_messages: Sequence[BaseMessage]
    notes: list[str]
    raw_notes: list[str]
    evidence_ledger: list[EvidenceRecord]
    final_report: str = ""


def today_utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def latest_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            return extract_text_content(getattr(message, "content", "")).strip()
    return ""


def human_texts(messages: list[Any]) -> list[str]:
    """Return all human messages in chronological order."""
    return [
        extract_text_content(getattr(message, "content", "")).strip()
        for message in messages
        if getattr(message, "type", "") == "human"
    ]


def normalize_topic_token(token: str) -> str:
    """Normalize a token for lightweight intent comparison."""
    cleaned = re.sub(r"[^a-z0-9']+", "", token.lower())
    if len(cleaned) <= 2 or cleaned in STOP_TOKENS:
        return ""
    if cleaned.endswith("ies"):
        cleaned = cleaned[:-3] + "y"
    elif cleaned.endswith("ing") and len(cleaned) > 5:
        cleaned = cleaned[:-3]
    elif cleaned.endswith("ed") and len(cleaned) > 4:
        cleaned = cleaned[:-2]
    elif cleaned.endswith("es") and len(cleaned) > 4:
        cleaned = cleaned[:-2]
    elif cleaned.endswith("s") and len(cleaned) > 3:
        cleaned = cleaned[:-1]
    return cleaned


def tokenize_for_intent(text: str) -> set[str]:
    """Tokenize normalized user text for lightweight intent comparison."""
    return {
        token
        for token in (normalize_topic_token(raw) for raw in re.findall(r"[A-Za-z0-9']+", text))
        if token
    }


def should_recheck_intent_on_follow_up(messages: list[Any]) -> bool:
    """Decide whether a follow-up turn likely shifts intent."""
    messages_text = [text for text in human_texts(messages) if text]
    if len(messages_text) < 2:
        return False

    latest = messages_text[-1]
    previous = messages_text[-2]
    latest_tokens = tokenize_for_intent(latest)
    previous_tokens = tokenize_for_intent(previous)
    if not latest_tokens or not previous_tokens:
        return False

    overlapping_tokens = latest_tokens & previous_tokens
    if overlapping_tokens:
        # Only force a shift recheck on explicit switch language. Common words like
        # "new" should not trigger clarification if topic overlap is clear.
        if latest_tokens & STRONG_FOLLOW_UP_SHIFT_MARKERS:
            return True
        return False
    if any(marker in latest_tokens for marker in FOLLOW_UP_CONTINUATION_MARKERS):
        return False
    if latest_tokens & FOLLOW_UP_SHIFT_MARKERS:
        return True

    return True


def _joined_human_text(messages: list[Any]) -> str:
    return " ".join(text.lower() for text in human_texts(messages) if text)


def is_broad_scope_request(messages: list[Any]) -> bool:
    """Heuristic for broad asks that should be scoped before research starts."""
    text = _joined_human_text(messages)
    if not text:
        return False
    has_broad_noun = any(marker in text for marker in _BROAD_SCOPE_NOUN_MARKERS)
    if not has_broad_noun:
        return False
    return any(marker in text for marker in _BROAD_SCOPE_ACTION_MARKERS)


def has_scope_boundary(messages: list[Any]) -> bool:
    """Return whether user messages include a concrete research boundary."""
    text = _joined_human_text(messages)
    if not text:
        return False

    has_timeframe = (
        bool(_SCOPE_YEAR_PATTERN.search(text))
        or bool(_SCOPE_QUARTER_PATTERN.search(text))
        or bool(_SCOPE_MONTH_PATTERN.search(text))
        or any(marker in text for marker in _SCOPE_TIMEFRAME_MARKERS)
    )
    if has_timeframe:
        return True

    has_geography = any(marker in text for marker in _SCOPE_GEOGRAPHY_MARKERS)
    if not has_geography and re.search(r"\bus\b", text):
        has_geography = True
    if has_geography:
        return True

    return any(marker in text for marker in _SCOPE_UNIVERSE_MARKERS)


def state_text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def normalize_note_list(value: Any) -> list[str]:
    """Normalize a state value into a clean list of note strings."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            text = state_text_or_none(item)
            if text:
                normalized.append(text)
        return normalized
    text = state_text_or_none(value)
    return [text] if text else []


def normalize_evidence_ledger(value: Any) -> list[EvidenceRecord]:
    """Normalize evidence ledger values into validated EvidenceRecord items."""
    if value is None:
        return []

    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[EvidenceRecord] = []
    for item in items:
        if isinstance(item, EvidenceRecord):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            try:
                normalized.append(EvidenceRecord.model_validate(item))
            except ValidationError:
                continue
            continue
    return normalized


def join_note_list(values: Any) -> str | None:
    """Join normalized notes into one synthesis-ready text block."""
    notes = normalize_note_list(values)
    if not notes:
        return None
    return "\n\n".join(notes).strip()


def is_token_limit_error(exc: Exception) -> bool:
    """Best-effort check for context/token limit failures across providers."""
    message = str(exc).lower()
    token_markers = (
        "context length",
        "maximum context",
        "too many tokens",
        "token limit",
        "max tokens",
    )
    return any(marker in message for marker in token_markers)


def latest_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            return message
    return None


def stringify_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        lines: list[str] = []
        for item in output:
            text = extract_text_content(item).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
    if isinstance(output, dict):
        if "content" in output:
            return extract_text_content(output.get("content", "")).strip()
        try:
            return json.dumps(output, ensure_ascii=True)
        except TypeError:
            return str(output)
    return extract_text_content(getattr(output, "content", output)).strip()


def compress_note_text(raw_text: str | None) -> str | None:
    text = state_text_or_none(raw_text)
    if not text:
        return None

    max_bullets = get_supervisor_notes_max_bullets()
    word_budget = get_supervisor_notes_word_budget()

    candidates: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"\n\n+", text):
        normalized = re.sub(r"\s+", " ", block).strip()
        if not normalized:
            continue
        dedupe_key = normalized.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(normalized)
        if len(candidates) >= max_bullets:
            break

    if not candidates:
        return None

    compressed = "\n".join(f"- {item}" for item in candidates)
    words = compressed.split()
    if len(words) > word_budget:
        compressed = " ".join(words[:word_budget]).strip() + " ..."
    return compressed.strip()
