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

Runtime is a native LangGraph graph (intake → supervisor → report) with the researcher leaf node powered by `create_deep_agent()`:

- Main graph: `build_app()` in `graph.py` — intake, supervisor subgraph, final report generation.
- Supervisor subgraph: hand-built LangGraph loop that dispatches `ConductResearch` tool calls.
- Researcher subgraph: `build_researcher_subgraph()` returns a `create_deep_agent()` compiled graph.
  - Accepts `{"messages": [HumanMessage(content=topic)]}` and returns `MessagesState`.
  - Built-in middleware: `SummarizationMiddleware`, `FilesystemMiddleware`, `PatchToolCallsMiddleware`, `TodoListMiddleware`, `AnthropicPromptCachingMiddleware`.
  - Uses the project's existing research tools (`search_web`, `fetch_url`, `think_tool`).
  - Post-processed via `extract_research_from_messages()` to produce compressed notes.

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
