<!--
Working notes. Keep this file decision-oriented and implementation-focused.
-->

# Exa Notes — Planner/Research Compiler + Search Integration

## Their Approach vs Ours (Side-by-Side)

| Exa deep research pattern (as described) | This repo’s pattern |
|---|---|
| **Planner**: analyze query, create parallel task specs | **Planner phase (orchestrator)**: analyze query complexity, create 1–4 parallel task specs, record via `write_todos` |
| **Parallel Tasks**: run tasks concurrently | **Parallel `task()` calls**: `task(subagent_type="research-agent", description=...)` |
| **Observer / loop**: check results, identify gaps, spawn more tasks | **Research Compiler phase (orchestrator)**: coverage checklist + (at most) one targeted follow-up wave |
| **Writer**: final synthesis | **Synthesize (orchestrator)**: final response directly in chat with citations |

What we adopted:
- Explicit task-spec planning before delegation (scope, evidence targets, output format, success criteria).
- A structured post-research coverage check that can loop back for targeted follow-ups.
- “Snippets-first” retrieval discipline.

What we changed (and why):
- We keep a **single orchestrator agent** (Planner/Compiler are phases), because the runtime is `deepagents.create_deep_agent()` and we want one canonical path pre-publication.
- We do **not** add a separate Writer agent or custom LangGraph topology.

## Exa Search Integration Notes

### Tool selection
- Provider priority is now **Exa → Tavily → None**.
- Exa is used when `EXA_API_KEY` is set and `langchain_exa` is available.
- Tavily remains a fallback when Exa is unavailable.

### Invocation defaults (snippets-first)
We configure Exa primarily at **invocation time** (inside `search_web`), defaulting to:
- `num_results`: `5`
- `highlights`: `True` (token-efficient evidence snippets)
- `type`: `"auto"`

We **do not** enable Exa’s `summary` by default (keeps preprocessing deterministic/LLM-free).

### Output shape gotchas
`ExaSearchResults` returns a list of dicts (commonly including `url`, `title`, `highlights`, `text`, etc.). Our search preprocessing remains deterministic:
1) normalize → 2) dedupe → 3) stable sort → 4) truncate/format.

Snippet extraction prefers:
1) `highlights` (joined) → 2) `raw_content` → 3) `content` → 4) `text`.

### Environment note: docs index
The Exa docs index (`https://exa.ai/docs/llms.txt`) could not be fetched in this environment (DNS/network restricted). Integration was implemented against the invocation contract documented in the task description and should be validated against the installed `langchain-exa` version.

## What We’re Deferring (Needs More API Surface)

1) **Exa `text_contents_options` on-demand**
   - Goal: let `search_web` optionally return full markdown page text via Exa instead of using `fetch_url`.
   - Blocker: `search_web`’s public interface is intentionally `search_web(query: str)`; adding options would change the interface.
   - Possible unblock: a new tool (e.g. `search_web_full`) or a deepagents-level tool-arg routing mechanism.

2) **Structured output schemas for subagents**
   - Goal: enforce a task-spec-shaped output (or evidence log schema) at the subagent definition level.
   - Blocker: requires verified `deepagents.create_deep_agent()` / subagent API support.

3) **Context filtering / sibling isolation**
   - Goal: prevent tasks from seeing sibling intermediate results when not needed.
   - Blocker: requires deepagents-level configuration support.

## Open Questions

- Confirm `langchain_exa.ExaSearchResults` constructor parameter naming across versions (`exa_api_key` vs `api_key`).
- Confirm exact `highlights` object schema (strings vs objects w/ `text` + scores).
- If we ever need domain/date filters, decide how to expose them without changing `search_web(query: str)`’s interface.
