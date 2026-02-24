# Deep Research

Open source deep research agent built on [LangGraph](https://github.com/langchain-ai/langgraph). Give it a question, it clarifies scope if needed, delegates focused research tasks in parallel, and returns a cited synthesis report.

## Architecture

```
intake -> clarify (if needed) -> research brief -> supervisor -> [researcher x N] -> final report
```

- **Intake** decides whether to ask one clarifying question or proceed directly.
- **Supervisor** breaks the brief into independent research units and dispatches them via `ConductResearch` tool calls. Multiple calls in one turn execute in parallel.
- **Researcher** is a deep agent with built-in middleware for context window management, large result eviction, and malformed tool call recovery. Uses `search_web`, `fetch_url`, and `think_tool`.
- **Final report** synthesizes compressed notes into a structured response with inline citations and a sources section.

## Quickstart

Python 3.11+ required.

```bash
git clone https://github.com/Hmbown/deepresearch.git
cd deepresearch
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and one search key
```

Run a query:

```bash
deepresearch "What are the latest advances in quantum error correction?"
```

Or use the module directly:

```bash
python -m deepresearch.cli "Compare retrieval strategies for production RAG systems"
```

Preflight check (validates model and search provider config without running a query):

```bash
python -m deepresearch.cli --preflight
```

## LangGraph Studio

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

Opens the Studio UI for interactive use with visual graph traces.

## Configuration

### Models

Supports any provider via `init_chat_model()`. Configure with `provider:model` strings.

| Variable | Default | Role |
|---|---|---|
| `ORCHESTRATOR_MODEL` | `openai:gpt-5.2` | Supervisor planning, compression, final report |
| `SUBAGENT_MODEL` | `openai:gpt-5.2` | Delegated research execution |

### Search

| Variable | Default | Notes |
|---|---|---|
| `SEARCH_PROVIDER` | `exa` | `exa`, `tavily`, or `none` |
| `EXA_API_KEY` | — | Required when `SEARCH_PROVIDER=exa` |
| `TAVILY_API_KEY` | — | Required when `SEARCH_PROVIDER=tavily` |

### Runtime Knobs

| Variable | Default | Description |
|---|---|---|
| `MAX_CONCURRENT_RESEARCH_UNITS` | `4` | Parallel researcher invocations per supervisor step |
| `MAX_RESEARCHER_ITERATIONS` | `6` | Total research units the supervisor can dispatch |
| `MAX_REACT_TOOL_CALLS` | `6` | Hard cap on tool calls per researcher invocation |
| `ENABLE_RUNTIME_EVENT_LOGS` | `false` | Structured runtime event logging |

See `.env.example` for the full set of options.

## Multi-turn Usage

Supports conversational follow-ups. Reuse the same `thread_id` and pass previous messages:

```python
import asyncio
from deepresearch.cli import run

async def main():
    thread_id = "demo-thread"
    messages = []
    for turn in [
        "Research the current state of nuclear fusion energy.",
        "Focus specifically on private sector investments since 2023.",
        "Now compare the leading companies by funding raised.",
    ]:
        result = await run(turn, thread_id=thread_id, prior_messages=messages)
        messages = list(result.get("messages", messages))

asyncio.run(main())
```

## Output Format

Reports are returned directly in chat. Each report includes section headings, inline citations (`[1]`, `[2]`) tied to specific claims, a sources section mapping citations to URLs, and explicit notes on uncertainty.

## Project Structure

```
src/deepresearch/
  graph.py                  # Main LangGraph runtime assembly
  researcher_subgraph.py    # Deep agent researcher + output extraction
  supervisor_subgraph.py    # Supervisor loop with parallel dispatch
  intake.py                 # Clarification routing + research brief
  report.py                 # Final report synthesis
  nodes.py                  # Research tools + search post-processing
  prompts.py                # All prompt templates
  config.py                 # Model, search, and runtime config
  cli.py                    # CLI entry point
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Online Evaluations

The repository includes an online LLM-as-judge eval harness for LangSmith traces.

Run locally:

```bash
python scripts/run_online_evals.py --project deepresearch-local --since 24h --limit 50
```

Run in GitHub Actions:

- Use the `online-evals` workflow via manual `workflow_dispatch`.
- Inputs: `project`, `since`, and `limit`.
- Output summary is printed directly in workflow logs.

## License

MIT
