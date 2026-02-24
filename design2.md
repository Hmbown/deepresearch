# Deep Research Agent — Design Decisions

This document captures design decisions as they evolve.
Sections 1–13 describe the current enforced architecture.
The changelog at the end records what changed and why.

---

## 1. Single Runtime Path

The system has exactly one execution path:

```
scope_intake -> research_supervisor -> final_report_generation
```

- `scope_intake` returns a LangGraph `Command` to route:
  - `goto="__end__"` when clarification is needed (pipeline stops; user re-enters).
  - `goto="research_supervisor"` when scope is clear (pipeline continues).
- `research_supervisor` always flows to `final_report_generation` via a static edge.
- `final_report_generation` always flows to `END`.

No dual pipelines, no conditional synthesis bypass, no fallback routing.
The graph is built in `graph.py:build_app()` and compiled once as the module-level `app`.

## 2. CLI/Runtime Parity Contract

**Invariant:** The CLI renders runtime truth exclusively. It never infers supervisor policy from message text, tool call counts, or string parsing.

### How it works

1. `supervisor_tools` (the runtime node) emits a `runtime_progress` dict as part of its state output. This payload includes:
   - `supervisor_iteration`, `dispatched_research_units`, `skipped_research_units`
   - `quality_gate_status` ("none" | "pass" | "retry"), `quality_gate_reason`
   - `evidence_record_count`, `source_domain_count`, `source_domains`
   - `research_units` — per-unit summaries with status, topic, duration, failure_reason

2. The CLI's `ProgressDisplay._handle_supervisor_runtime_progress()` reads this payload on every `supervisor_tools` chain-end event. All wave dispatch messaging, quality gate rendering, and per-unit failure summaries derive exclusively from this payload.

3. If `runtime_progress` is absent (e.g. the event lacks the key), the CLI emits no dispatch/gate/wave text. It does not fall back to parsing `supervisor_messages`.

### What the CLI reads directly from the event stream

- **Node names** (e.g. `"scope_intake"`, `"supervisor_tools"`, `"final_report_generation"`) for phase labeling.
- **Researcher chain input** for topic display (the actual HumanMessage content sent to each researcher).
- **Researcher chain output** for per-unit evidence stats (evidence_ledger and messages from the researcher's own output).
- **Tool events** (search_web, fetch_url, think_tool) for verbose tool-level display.

These are all reading actual runtime data, not inferring supervisor decisions.

### Parity guard tests

- `test_progress_display_no_dispatch_or_gate_output_when_runtime_progress_absent` — verifies CLI emits nothing when only supervisor_messages are present.
- `test_progress_display_does_not_derive_dispatch_count_from_tool_call_count` — verifies dispatch count comes from `runtime_progress`, not tool message count.
- `test_progress_display_ignores_supervisor_tool_message_text_when_runtime_progress_present` — verifies runtime_progress overrides misleading message text.

## 3. Quality Gate

The supervisor does not decide when research is complete. A deterministic evidence quality gate does.

### Gate evaluation (`_evaluate_research_complete_gate`)

When the supervisor issues `ResearchComplete`, the runtime checks:

| Check                        | Threshold (default, env-configurable) |
|------------------------------|---------------------------------------|
| Total evidence records       | >= 5 (`MIN_EVIDENCE_RECORDS`)         |
| Cited evidence records       | >= 5 (records with source_urls)       |
| Distinct source domains      | >= 3 (`MIN_SOURCE_DOMAINS`)           |

- **Pass:** Research proceeds to synthesis.
- **Reject:** A `ToolMessage` with `[ResearchComplete rejected: ...]` is returned to the supervisor, and `quality_gate_status` is set to `"retry"` in `runtime_progress`. The supervisor can issue more research.

### Gate output contract

The gate result is serialized into `runtime_progress`:
```python
"quality_gate_status": "pass" | "retry" | "none",
"quality_gate_reason": "<rejection reason or None>",
```
The CLI reads these fields. It never parses `[ResearchComplete ...]` strings.

## 4. Evidence Model

### EvidenceRecord (Pydantic)

```python
class EvidenceRecord(BaseModel):
    claim: str                              # Atomic factual claim
    source_urls: list[str]                  # Supporting URLs
    confidence: float                       # Deterministic, always 0.5
    contradiction_or_uncertainty: str | None # Flagged uncertainty
```

- Confidence is always 0.5 (deterministic, model-agnostic).
- Uncertainty is flagged by regex pattern matching, not LLM judgment.
- Evidence records are extracted by `extract_research_from_messages()` using citation parsing, not LLM calls.

### Evidence lifecycle

1. Researcher produces MessagesState with tool outputs and a final AI synthesis.
2. `extract_research_from_messages()` post-processes into: compressed notes, raw notes, evidence records.
3. Evidence records accumulate in `SupervisorState.evidence_ledger` via additive reducer.
4. Quality gate evaluates the accumulated ledger.
5. Final report synthesis uses notes + evidence for citation-backed output.

## 5. Supervisor Subgraph

Hand-built LangGraph loop: `supervisor -> supervisor_tools -> supervisor (loop) | END`.

### supervisor node

- Calls the orchestrator LLM with bound tools: `ConductResearch`, `ResearchComplete`, `think_tool`.
- Receives notes context (compressed + raw) from prior iterations.
- Returns AI message with tool calls.

### supervisor_tools node

- Dispatches `ConductResearch` calls to `build_researcher_subgraph()` instances in parallel.
- Respects hard caps: `MAX_CONCURRENT_RESEARCH_UNITS`, `MAX_RESEARCHER_ITERATIONS`.
- Evaluates quality gate on `ResearchComplete` calls.
- Emits `runtime_progress` with full dispatch/gate/unit summary.
- Sanitizes researcher failure messages (no internal URLs or stack traces).

### Routing

- `route_supervisor`: continues to `supervisor_tools` if AI has tool calls and iteration cap not reached.
- `route_supervisor_tools`: returns to `supervisor` unless iteration cap reached.

## 6. Researcher Subgraph

Built via `deepagents.create_deep_agent()` with:
- Model: `SUBAGENT_MODEL` (default `openai:gpt-5.2`)
- Tools: `search_web`, `fetch_url`, `think_tool`
- System prompt: `RESEARCHER_PROMPT` or `RESEARCHER_PROMPT_NO_SEARCH`

### No-search mode

When `SEARCH_PROVIDER=none`, the researcher prompt switches to `RESEARCHER_PROMPT_NO_SEARCH` which:
- Removes search_web from the tool list and prompt.
- Only formats the keys used by that template (`max_react_tool_calls`).
- Explicitly tells the researcher not to attempt search calls.

### Search preprocessing pipeline

Deterministic, LLM-free: `normalize -> deduplicate -> sort -> truncate -> format`.

- Supports multiple provider shapes: dict results, SearchResponse objects, string fallback.
- Deduplicates by URL (or by content fingerprint if no URL).
- Sorts deterministically by URL then title.
- Truncates to `MAX_SEARCH_RESULTS_FOR_AGENT` (8).
- Emits `search_preprocess` metrics with `llm_calls_in_preprocess: 0`.

### fetch_url safety

- SSRF protection: blocks private IPs, localhost, .local, .internal.
- Scheme validation: only http/https.
- Error sanitization: no internal IPs or hostnames in error messages.
- Content extraction: trafilatura -> BeautifulSoup fallback -> error.
- Truncation: max 8000 chars.

## 7. Error Sanitization

### Supervisor level (`_sanitize_research_unit_failure`)

Exception types map to safe messages:
- `TimeoutError` -> "research unit timed out"
- Recursion limit match -> preserved as "Recursion limit of N reached"
- Everything else -> "research unit execution failed"

No internal URLs, stack traces, or provider-specific error details leak into tool messages.

### CLI level (`_summarize_research_failure`)

Recursion limit text is rewritten for users: "hit tool call limit (N steps)".
Other failures are truncated to 180 chars.
`For troubleshooting, visit:` URLs are stripped.

### fetch_url level (`_format_fetch_error`)

Provider-specific exceptions map to generic messages:
- Timeout -> "request timed out"
- HTTP status -> "remote server returned HTTP {code}"
- Network -> "network error while fetching URL"

## 8. State Models

### ResearchState (main graph)

```
messages, research_brief, intake_decision, awaiting_clarification,
supervisor_messages, notes, raw_notes, evidence_ledger, final_report
```

- `intake_decision`: "clarify" | "proceed" — set by scope_intake.
- `awaiting_clarification`: bool — tracks whether user input is pending.
- `supervisor_messages`, `notes`, `raw_notes` use overwrite semantics (not additive reducers) at the main graph level to prevent stale state accumulation across checkpointed turns.

### SupervisorState (subgraph)

```
supervisor_messages, research_brief, notes, raw_notes,
evidence_ledger, research_iterations, runtime_progress
```

- Uses additive reducers for `notes`, `raw_notes`, `evidence_ledger`, `supervisor_messages` within the supervisor loop.
- `research_iterations`: tracks total dispatched units for cap enforcement.
- `runtime_progress`: dict emitted by supervisor_tools for CLI consumption.

## 9. Intake Scoping

### First turn

1. `ClarifyWithUser` structured output: decides clarify vs proceed.
2. If clarify: returns question to user, sets `intake_decision="clarify"`, pipeline stops.
3. If proceed: generates `ResearchBrief` structured output, hands off to supervisor.

### Follow-up turns

- If prior `intake_decision="proceed"` and follow-up looks like same topic: skip clarification, regenerate brief.
- If follow-up triggers topic shift (detected by `should_recheck_intent_on_follow_up`): re-run clarification. On topic shift + clarify, prior supervisor state is hard-reset (notes, evidence, supervisor_messages all cleared).

### Topic shift detection

Lightweight, token-based:
- Tokenize + normalize both latest and previous human messages.
- If token overlap exists: only shift on explicit markers ("instead", "switch", "different").
- If no overlap: shift unless continuation markers present ("also", "expand", "deeper").

## 10. Final Report Synthesis

- Synthesizes from compressed notes + raw notes + evidence ledger.
- Retries on token limit errors with progressively smaller note subsets (up to 3 attempts).
- Falls back to compressed notes -> raw notes -> default message if LLM fails.
- Strips internal meta lines (ConductResearch, tool_call_id, etc.).
- Ensures source transparency: appends Sources section with cited URLs from evidence ledger if missing.

## 11. Configuration

All runtime knobs are env-var driven with safe defaults:

| Variable                         | Default       | Purpose                               |
|----------------------------------|---------------|---------------------------------------|
| `ORCHESTRATOR_MODEL`             | openai:gpt-5.2| Orchestration and synthesis model     |
| `SUBAGENT_MODEL`                 | openai:gpt-5.2| Delegated research model              |
| `SEARCH_PROVIDER`                | exa           | Search backend (exa/tavily/none)      |
| `MAX_CONCURRENT_RESEARCH_UNITS`  | 6             | Parallel researcher cap per wave      |
| `MAX_RESEARCHER_ITERATIONS`      | 60            | Safety ceiling for total dispatches   |
| `MAX_REACT_TOOL_CALLS`           | 40            | Soft tool call guidance per researcher|
| `RESEARCHER_SEARCH_BUDGET`       | 15            | Soft search call budget per researcher|
| `MIN_EVIDENCE_RECORDS`           | 5             | Quality gate: min evidence records    |
| `MIN_SOURCE_DOMAINS`             | 3             | Quality gate: min source domains      |
| `SUPERVISOR_NOTES_MAX_BULLETS`   | 40            | Max compressed note bullets           |
| `SUPERVISOR_NOTES_WORD_BUDGET`   | 1200          | Max compressed note words             |
| `ENABLE_ONLINE_EVALS`            | false         | Online LLM-as-judge evals            |
| `EVAL_MODEL`                     | openai:gpt-4.1-mini | Judge model for evals           |

## 12. Environment Bootstrap

- `.env` loaded once per process via `bootstrap_env()` in `__init__.py`.
- Discovery priority: `DEEPRESEARCH_ENV_FILE` override -> walk up from cwd for `.env` -> walk up for `.env.example` -> walk up for `pyproject.toml` with project name.
- `ensure_runtime_env_ready()` validates required keys and search provider config.
- `runtime_preflight()` runs full diagnostic: dotenv, runtime keys, search provider, deepagents dependency, LangSmith auth.
- `run_setup_wizard()` provides interactive first-time setup.

## 13. Online Evaluations

Optional LLM-as-judge framework:
- `eval_answer_quality`: judges final report quality.
- `eval_process_quality`: judges research process (search strategy, source diversity).
- `eval_composite`: weighted combination (60% answer, 40% process).
- Runs asynchronously via `OnlineEvalCallbackHandler` on root chain completion.
- Disabled by default (`ENABLE_ONLINE_EVALS=false`).

---

## 14. Design Notes (Author)

- After getting the agent working as an MVP, I compared output against ChatGPT 5.2 (GPT subscription) on the same question.
- The comparison made it obvious I was limiting the agent's capabilities with overly strict tool-call limits and search limits.
- Current takeaway: the more artificial constraints I put on the agent, the worse the output tends to be. Finding a happy medium that keeps costs and focus under control without kneecapping quality.
- Architecture choice: normal graphs/agents for everything except the research supervisor.
- The research supervisor is a DeepAgent — it gets default memory compaction and the tool-calling abilities that come with DeepAgents.
- The supervisor spawns sub-agents via tool calls and then synthesizes the final report.
- Evals are set up but not useful yet — still need manual tuning and to roll back early restrictions to reach target quality.

---

## Changelog

### 2026-02-24 — Depth unlocking, prompt rewrite, plan checkpoint

**Problem:** The system was silently killing deep research runs. Users saw "hit tool call limit (41 steps)" after ~2 waves because internal defaults were set too conservatively. Output was thin — far fewer sources and citations than comparable systems.

**Root cause:** `MAX_RESEARCHER_ITERATIONS` (16) was the critical bottleneck. With 6 concurrent units per wave, the system exhausted its total dispatch budget after ~2-3 waves. Other limits (search budget, tool calls, note compression) compounded the issue.

#### Config defaults raised

| Variable                        | Before | After | Rationale                                           |
|---------------------------------|--------|-------|-----------------------------------------------------|
| `RESEARCHER_SEARCH_BUDGET`      | 8      | 15    | Was too low for multi-angle evidence gathering       |
| `MAX_REACT_TOOL_CALLS`          | 20     | 40    | Researchers hit the wall mid-investigation           |
| `MAX_RESEARCHER_ITERATIONS`     | 16     | 60    | Critical bottleneck — only allowed ~2 waves of 6    |
| `SUPERVISOR_NOTES_MAX_BULLETS`  | 20     | 40    | Notes were getting truncated, losing evidence        |
| `SUPERVISOR_NOTES_WORD_BUDGET`  | 500    | 1200  | Same — compressed notes were too small for coverage  |

#### Quality gate made configurable

| Threshold                  | Before       | After                                   |
|----------------------------|--------------|-----------------------------------------|
| Min evidence records       | 2 (hardcoded)| 5 (env: `MIN_EVIDENCE_RECORDS`)         |
| Min source domains         | 2 (hardcoded)| 3 (env: `MIN_SOURCE_DOMAINS`)           |

Previously these were constants in `supervisor_subgraph.py`. Now they read from env vars at module load, so operators can tune them without code changes. The progress payload also reports the thresholds so the CLI (or any consumer) knows what the gate is checking against.

#### Claim extraction cap raised

`researcher_subgraph.py`: max extracted evidence claims per researcher run: 12 → 25. The old cap was discarding valid evidence from thorough researchers.

#### All prompts rewritten

The prior prompts were written by a non-Claude AI and produced robotic, over-constrained agent behavior. The rewrite:

- **SUPERVISOR_PROMPT**: From bureaucratic ("You are a Research Supervisor orchestrating deep research...") to conversational ("You are the lead research supervisor. Your job is to break down a research question into focused tracks..."). Encourages 3-5 waves explicitly. Removes "minimum set" language that made the supervisor stop too early.
- **RESEARCHER_PROMPT**: From template-heavy ("Search budget guidance: ...searches. Evidence targets: ...") to natural guidance ("You have {budget} search calls — that's plenty. Use what you need."). Removed self-limiting language.
- **RESEARCHER_PROMPT_NO_SEARCH**: Simplified to match the search-enabled version's tone, minus search references.
- **FINAL_REPORT_PROMPT**: More direct, stronger citation requirements, explicit instructions to distinguish well-established vs preliminary findings.

The key insight: prompts that read like natural instructions to a colleague produce better agent behavior than prompts that read like configuration files.

#### Plan checkpoint added to intake

Before: After scoping, the system showed a hardcoded confirmation message with domain-specific boilerplate ("bankruptcy-risk signal strength" etc.) regardless of the actual topic.

After: Intake generates a `ResearchPlan` (structured output with scope, research tracks, evidence strategy, output format) via the orchestrator LLM, then formats it for user review. If plan generation fails, falls back to showing the research brief directly. The plan is topic-specific and gives the user a real preview of what 20 minutes of research will cover before they approve.

New `RESEARCH_PLAN_PROMPT` added to `prompts.py`. New `ResearchPlan` Pydantic model added to `state.py`.

### 2026-02-24 — Decouple claim cap and source cap in evidence extraction

**Problem:** Evidence extraction used a single hard-coded cap (25 claims per researcher) but had no independent control over how many source URLs each claim could retain. Operators could not tune claim volume vs source breadth independently.

**What changed:**

Two new env-configurable knobs added to `config.py`:

| Variable                              | Default | Purpose                                      |
|---------------------------------------|---------|----------------------------------------------|
| `MAX_EVIDENCE_CLAIMS_PER_RESEARCH_UNIT` | 5       | Cap on claims extracted per researcher result |
| `MAX_SOURCE_URLS_PER_CLAIM`           | 5       | Cap on source URLs retained per claim         |

`MAX_CONCURRENT_RESEARCH_UNITS` default was also lowered from 6 to 4 (already in code; design doc section 11 table previously said 6 — now corrected).

**How they are wired:**

- `researcher_subgraph.py:_extract_claim_lines()` reads `get_max_evidence_claims_per_research_unit()` to cap claim extraction.
- `researcher_subgraph.py:_extract_evidence_records()` reads `get_max_source_urls_per_claim()` to independently cap URLs per claim, including citation-resolved, inline, and fallback URLs.
- Both caps are read at call time via `_resolve_int_env()`, not at module load, so env overrides take effect immediately.
- `.env.example` documents both knobs.

**Architecture:** No new files, no new abstractions, no dual pipelines. Two new getter functions in `config.py`, two call sites in `researcher_subgraph.py`, tests in both focused test files.

**Test results:**

- `tests/test_config_and_cli.py`: 40/40 passed — covers default fallback, env override, and constant assertion for both knobs.
- `tests/test_evidence_quality_gate.py`: 8/8 passed — includes `test_extract_evidence_records_respects_max_claims_per_research_unit` which sets `MAX_EVIDENCE_CLAIMS_PER_RESEARCH_UNIT=2` and asserts only 2 records are returned from 4-claim input.
- Full suite: 174/174 passed, 1 skipped (SQLite checkpointer env-specific).
- CLI entry point verified: `python3 -m deepresearch.cli --help` runs successfully.

**Deployment status:** Committed and pushed to `main`.

**Remaining risks / next steps:**

- Section 11 table in this doc should be updated to include `MAX_EVIDENCE_CLAIMS_PER_RESEARCH_UNIT` and `MAX_SOURCE_URLS_PER_CLAIM` rows and correct `MAX_CONCURRENT_RESEARCH_UNITS` default from 6 to 4 — deferred to next doc pass.
- No integration test yet that exercises both caps simultaneously in a live researcher run (unit tests cover the extraction logic in isolation).
- The claim cap default of 5 is conservative; may need tuning upward once live output quality is assessed.
