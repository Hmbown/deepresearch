# deepresearch

Deep research agent built on native LangGraph subgraphs with one canonical runtime path:

`route_turn -> clarify_with_user -> write_research_brief -> research_supervisor -> final_report_generation`

## Prerequisites

- Python `>=3.11`
- `pip`
- OpenAI API key (`OPENAI_API_KEY`) for default models
- Optional: Exa API key (`EXA_API_KEY`) for higher-quality web search
- Optional: LangSmith keys for tracing (`LANGCHAIN_API_KEY` or `LANGSMITH_API_KEY`)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
[ -f .env ] || cp .env.example .env
```

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Required provider key for default orchestrator/subagent models |
| `EXA_API_KEY` | No | — | Enables Exa search tool in researcher runs |
| `ORCHESTRATOR_MODEL` | No | `openai:gpt-5.2` | Intake + supervisor + final synthesis model |
| `SUBAGENT_MODEL` | No | `openai:gpt-5.2` | Researcher subagent model |
| `MAX_STRUCTURED_OUTPUT_RETRIES` | No | `3` | Retry budget for structured intake/brief schemas |
| `MAX_REACT_TOOL_CALLS` | No | `6` | Max tool calls per researcher unit |
| `MAX_CONCURRENT_RESEARCH_UNITS` | No | `4` | Max parallel `ConductResearch` units per wave |
| `MAX_RESEARCHER_ITERATIONS` | No | `6` | Max total `ConductResearch` units per supervisor run |
| `SUPERVISOR_NOTES_MAX_BULLETS` | No | `10` | Note compression cap |
| `SUPERVISOR_NOTES_WORD_BUDGET` | No | `250` | Note compression word budget |
| `SUPERVISOR_FINAL_REPORT_MAX_SECTIONS` | No | `8` | Final report structure cap hint |
| `RESEARCHER_SIMPLE_SEARCH_BUDGET` | No | `3` | Search budget for simple delegated tasks |
| `RESEARCHER_COMPLEX_SEARCH_BUDGET` | No | `5` | Search budget for complex delegated tasks |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing when set truthy |
| `LANGSMITH_TRACING` | No | — | Alternate tracing flag (also supported) |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |
| `LANGSMITH_API_KEY` | No | — | Alternate LangSmith key var |
| `LANGCHAIN_PROJECT` | No | `deepresearch-local` | LangSmith project name |
| `LANGSMITH_PROJECT` | No | — | Alternate project var |
| `LANGCHAIN_ENDPOINT` | No | `https://api.smith.langchain.com` | Optional custom LangSmith endpoint |

## Preflight

Run before first query:

```bash
source .venv/bin/activate
python -m deepresearch.cli --preflight
```

Interpretation:

- `dotenv_file=PASS`: `.env` exists.
- `runtime_keys=PASS`: required runtime keys are present (`OPENAI_API_KEY`).
- `search_key=PASS`: info check only (`EXA_API_KEY` optional).
- `langsmith=PASS`: either tracing is disabled, or tracing is enabled and auth works.
- Any `FAIL`: fix the reported key/config and rerun preflight.

## Quickstart

One-shot:

```bash
source .venv/bin/activate
python -m deepresearch.cli "Compare retrieval strategies for production RAG systems"
```

Interactive multi-turn:

```bash
source .venv/bin/activate
python -m deepresearch.cli
```

Programmatic two-turn continuity:

```python
import asyncio
from deepresearch.cli import run

async def main():
    thread_id = "example-thread"
    turn1 = await run("Please research this topic for me.", thread_id=thread_id)
    turn2 = await run(
        "Focus on U.S. and EU policy impacts from 2020-2026 with citations.",
        thread_id=thread_id,
        prior_messages=list(turn1.get("messages", [])),
    )
    print(turn1.get("intake_decision"), turn2.get("intake_decision"))

asyncio.run(main())
```

## Long-Running Runtime Validation

This is the release validation flow used to prove:

- turn 1 routes to clarification (`intake_decision=clarify`)
- turn 2 proceeds (`intake_decision=proceed`)
- fanout executes `ConductResearch` at least 3 times
- final report is synthesized
- LangSmith run presence is verified when tracing is enabled

Run:

```bash
source .venv/bin/activate
python scripts/validate_multiagent_runtime.py
```

Notes:

- The script force-sets:
  - `MAX_CONCURRENT_RESEARCH_UNITS=3`
  - `MAX_RESEARCHER_ITERATIONS=3`
- It prints:
  - `turn1.decision`
  - `turn1.clarification`
  - `turn2.decision`
  - `turn2.conduct_research_count`
  - `turn2.final_report_length`
  - `turn2.final_report_first_300`
  - LangSmith trace check status when enabled

## Testing

```bash
source .venv/bin/activate
python -m compileall src/deepresearch
python -m pytest -q
```

## LangSmith Troubleshooting

### Tracing disabled intentionally

If preflight says tracing skipped/disabled, this is expected when both `LANGCHAIN_TRACING_V2` and `LANGSMITH_TRACING` are not truthy.

### Missing LangSmith key

If tracing is enabled but key is missing, preflight fails with guidance. Set one of:

- `LANGCHAIN_API_KEY`
- `LANGSMITH_API_KEY`

### Invalid key or endpoint

If auth fails with tracing enabled:

1. Verify key value.
2. Verify `LANGCHAIN_ENDPOINT` (or remove it to use default).
3. Verify project name (`LANGCHAIN_PROJECT`/`LANGSMITH_PROJECT`).
4. Rerun `python -m deepresearch.cli --preflight`.

## LangSmith Placeholders

Replace these with real screenshots from your environment:

- Pipeline trace placeholder:

![LangSmith pipeline trace placeholder](docs/images/langsmith-trace-placeholder.png)

- LangSmith Studio setup placeholder:

![LangSmith Studio setup placeholder](docs/images/langsmith-studio-setup-placeholder.png)

## Architecture Notes

- Single canonical runtime path only.
- No dual pipelines or transition flags.
- Core behavior preserved: clarify when needed, delegate focused research, synthesize final answer.
- Runtime does not auto-write reports to disk.
