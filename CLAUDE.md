# Deep Research Agent

Deep research system built on `deepagents` with LangGraph runtime under the hood.

## Pre-GitHub Core Policy (Mandatory)

Until this repo is published as the reference implementation, keep architecture minimal:

1. Rewrite/remove over compatibility layers.
2. No dual pipelines or transition shims.
3. Use one canonical runtime path.
4. Remove stages that do not clearly improve quality.
5. Preserve core behavior: clarify (if needed) -> delegate focused research -> synthesize.
6. Defer non-core hardening unless explicitly requested.

## Project Structure

```text
deepresearch/
├── src/deepresearch/
│   ├── __init__.py
│   ├── graph.py       # create_deep_agent() assembly and exported `app`
│   ├── nodes.py       # reusable research tools + deterministic search processing
│   ├── prompts.py     # CLARIFY/BRIEF + SUPERVISOR_PROMPT + RESEARCHER_PROMPT
│   ├── config.py      # model + search provider config
│   └── cli.py         # CLI invoke path (messages-based)
├── tests/
├── langgraph.json
├── pyproject.toml
├── .env
└── .gitignore
```

## Architecture

Runtime is a single deep agent created via `create_deep_agent()`:

- Main orchestrator agent uses:
  - `write_todos` via middleware
  - `task()` subagent delegation via `SubAgentMiddleware`
  - summarization + filesystem middleware from deepagents defaults
- One specialized `research-agent` subagent is provided.
- Multiple `task()` calls in one model turn execute in parallel.

## Output Contract

- Default final output is a structured assistant response in chat.
- Do not auto-write report files.
- Use `write_file` only when the user explicitly requests export.

## Models and Providers

- `ORCHESTRATOR_MODEL` (default `openai:gpt-5.2`) for orchestration and synthesis.
- `SUBAGENT_MODEL` (default `openai:gpt-5.2`) for delegated research tasks.
- Provider routing via `init_chat_model()`.

## Search

- Provider priority: Exa -> Tavily -> None.
- Search preprocessing remains deterministic:
  - normalize -> dedupe -> sort -> truncate.
- `fetch_url` remains available for full-page extraction.
- `think_tool` remains available for disciplined research loops.

## Implementation Expectations

1. Prefer deletion over abstraction when both solve the problem.
2. Keep interface and behavior easy to reason about for new contributors.
3. Keep tests aligned with the deepagents architecture (no legacy graph-node assertions).

## Running

```bash
pip install -e .
deepresearch "Your research query"
python -m deepresearch.cli "Your research query"
langgraph dev
```
