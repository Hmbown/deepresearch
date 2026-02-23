# Deep Research Agent

Implementation handoff for contributors working in this repository.

## Pre-GitHub Core Policy (Mandatory)

Until this project is published as the reference implementation, optimize for a clean core:

1. Rewrite/remove over backward-compat layers.
2. No dual pipelines or transition flags.
3. Keep one canonical runtime path.
4. Remove stages that do not clearly justify their cost.
5. Preserve core behavior: clarify when needed, delegate focused research, synthesize final answer.
6. Defer non-core hardening tracks unless explicitly requested.

If in doubt, choose the simpler architecture.

## Project Structure

```text
deepresearch/
├── src/deepresearch/
│   ├── __init__.py
│   ├── graph.py
│   ├── nodes.py
│   ├── prompts.py
│   ├── config.py
│   └── cli.py
├── tests/
├── langgraph.json
├── pyproject.toml
├── .env
└── .gitignore
```

## Runtime Model

The app is created with native LangGraph subgraphs in `build_app()`:

- Main path: `route_turn -> clarify_with_user -> write_research_brief -> research_supervisor -> final_report_generation`.
- Supervisor subgraph delegates focused work via `ConductResearch` and can run researchers in parallel.
- Researcher subgraph runs a bounded tool loop using `search_web`, `fetch_url`, and `think_tool`, then compresses findings.
- Final report generation synthesizes supervisor notes into the user-facing response.

## Output Contract

- Default output is a well-organized response in chat.
- Do not auto-write reports to disk.
- Use `write_file` only if explicitly requested by the user.

## Models and Providers

- `ORCHESTRATOR_MODEL` (default `openai:gpt-5.2`): orchestration + synthesis.
- `SUBAGENT_MODEL` (default `openai:gpt-5.2`): subagent execution.
- Search provider in runtime: Exa.

## Search and Tools

Keep and preserve:

- Deterministic search preprocessing: normalize -> dedupe -> sort -> truncate.
- `fetch_url` extraction tool (`httpx` + `trafilatura` fallback).
- `think_tool` for strategic reflection in research loops.

## Architecture Decision (SHA-754): Single Runtime Path Only

- Decision: keep one canonical runtime path in `build_app()` with no dual pipelines, workflow alternates, or report-generation branches.
- Guardrail evidence:
  - `src/deepresearch/graph.py` contains one main route path (`route_turn -> clarify_with_user -> write_research_brief -> research_supervisor -> final_report_generation`).
  - `tests/test_architecture_guardrails.py` blocks legacy graph tokens and machine-specific path leakage.
- This is the canonical implementation rule until a concrete request justifies expansion.

## Deferred Work (Issues 752, 753)

- SHA-753 — quality/evaluation harness
  - Decision: defer.
  - Design note: keep current prompt-driven regression checks; postpone benchmarking/evaluator harness to a dedicated performance track.
  - Next steps (when explicitly resumed): add a small prompt-quality fixture set and a parallelism regression dataset.
  - Effort estimate: 8–16h.
  - Risk: medium-high (maintainable evaluation fixture drift + benchmark cost).

## Implementation Expectations

When changing architecture:

1. State the concrete problem and measurable expected gain.
2. Prefer deletion over abstraction when both solve it.
3. Keep behavior and interfaces straightforward for new contributors.
4. Update tests with every structural change.
