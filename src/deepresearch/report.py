"""Final report synthesis node."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from .config import get_llm, get_supervisor_final_report_max_sections
from .prompts import FINAL_REPORT_PROMPT
from .runtime_utils import invoke_runnable_with_config
from .state import (
    FALLBACK_FINAL_REPORT,
    ResearchState,
    compress_note_text,
    is_token_limit_error,
    join_note_list,
    normalize_note_list,
    state_text_or_none,
    stringify_tool_output,
    today_utc_date,
)

_logger = logging.getLogger(__name__)


async def final_report_generation(
    state: ResearchState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """Create final report from supervisor outputs when not already completed."""
    final_report = state_text_or_none(state.get("final_report"))
    note_chunks = normalize_note_list(state.get("notes"))
    raw_note_chunks = normalize_note_list(state.get("raw_notes"))

    if not final_report:
        model = get_llm("orchestrator")
        max_attempts = 3
        for attempt in range(max_attempts):
            if not note_chunks and raw_note_chunks:
                compressed_seed = compress_note_text(join_note_list(raw_note_chunks))
                if compressed_seed:
                    note_chunks = [compressed_seed]

            if attempt == 0:
                prompt_note_chunks = note_chunks
                prompt_raw_chunks = raw_note_chunks
            else:
                note_limit = max(1, len(note_chunks) // (attempt + 1)) if note_chunks else 0
                raw_limit = max(1, len(raw_note_chunks) // (attempt + 1)) if raw_note_chunks else 0
                prompt_note_chunks = note_chunks[:note_limit] if note_limit else []
                prompt_raw_chunks = raw_note_chunks[:raw_limit] if raw_limit else []

            notes_text = join_note_list(prompt_note_chunks) or "[No compressed notes]"
            raw_notes_text = join_note_list(prompt_raw_chunks) or "[No raw notes]"
            synthesis_prompt = "\n\n".join(
                [
                    FINAL_REPORT_PROMPT.format(
                        current_date=today_utc_date(),
                        final_report_max_sections=get_supervisor_final_report_max_sections(),
                    ),
                    f"Research brief:\n{state_text_or_none(state.get('research_brief')) or '[Missing brief]'}",
                    f"Compressed notes:\n{notes_text}",
                    f"Raw notes:\n{raw_notes_text}",
                    "Write the final report now.",
                ]
            )

            try:
                response = await invoke_runnable_with_config(model, [HumanMessage(content=synthesis_prompt)], config)
                final_report = state_text_or_none(stringify_tool_output(response))
                if final_report:
                    break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt + 1 < max_attempts and is_token_limit_error(exc):
                    continue
                _logger.exception("final_report_generation synthesis failed; using fallback text")
                break

    if not final_report:
        final_report = join_note_list(note_chunks) or join_note_list(raw_note_chunks) or FALLBACK_FINAL_REPORT

    return {
        "messages": [AIMessage(content=final_report)],
        "final_report": final_report,
        "intake_decision": "proceed",
        "awaiting_clarification": False,
    }
