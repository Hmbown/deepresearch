# Deep Research

LangGraph multi-agent system that clarifies scope, runs focused web research in parallel, and synthesizes a cited report.

![Architecture](docs/architecture.png)

**Pipeline:** `scope_intake` (clarify & brief) → `research_supervisor` (parallel researcher dispatch via Send) → `final_report_generation` (synthesis with citations)

---

## Quick Start

```bash
git clone https://github.com/Hmbown/deepresearch.git
cd deepresearch
pip install -e .
cp .env.example .env   # set OPENAI_API_KEY at minimum
deepresearch "your research question"
```

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| `OPENAI_API_KEY` | Required |
| `TAVILY_API_KEY` | Optional — default search provider (Exa also supported) |

## Configuration

```bash
cp .env.example .env
# Edit .env — set at minimum OPENAI_API_KEY
```

Run preflight to verify everything is wired up:

```bash
deepresearch --preflight
```

Pass a LangSmith project name to also verify tracing auth:

```bash
deepresearch --preflight my-langsmith-project
```

## Usage

### CLI

```bash
deepresearch "your research question"
```

### No-search fallback (smoke test)

Validate the full pipeline without web search:

```bash
SEARCH_PROVIDER=none deepresearch "summarize the key differences between Python 3.11 and 3.12"
```

### LangGraph Studio

Start the local dev server:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 \
  langgraph dev --host 127.0.0.1 --port 2024 --no-browser --allow-blocking
```

Then open Studio at: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

<details>
<summary>Graph in LangGraph Studio</summary>

![LangGraph Studio Graph](docs/langgraph-graph.png)

</details>

## Architecture

| Module | Lines | Role |
|---|---|---|
| `graph.py` | 41 | Pipeline assembly — wires intake → supervisor → report |
| `state.py` | 550 | Shared state models, Pydantic schemas, text normalization |
| `intake.py` | 293 | Scope clarification with Command routing |
| `supervisor_subgraph.py` | 709 | Supervisor loop with Send-based parallel fan-out |
| `researcher_subgraph.py` | 147 | Researcher agent with search/fetch/think tools |
| `report.py` | 331 | Final synthesis with retry logic and source transparency |
| `nodes.py` | 378 | Research tools — search, fetch (SSRF-protected), think |
| `prompts.py` | 209 | All prompt templates |
| `config.py` | 243 | Model/search provider configuration |
| `env.py` | 306 | Environment bootstrap and preflight checks |
| `cli.py` | 1063 | CLI with streaming ProgressDisplay |
| `runtime_utils.py` | 86 | Runnable invocation helpers |
| `message_utils.py` | 31 | Message content extraction |
| `evals/evaluators.py` | 318 | LLM-as-judge evaluation (answer + process quality) |
| `evals/callback.py` | 129 | Online eval callback handler |

## LangSmith Tracing

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=deepresearch-local
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `deepagents` not found | `pip install -e .` |
| Search credit errors (Exa `402`, Tavily quota) | `SEARCH_PROVIDER=none` in `.env` |
| LangSmith shows "No data" | Check billing/quota limits on free tier |
| OpenAI Responses API errors | Check `OPENAI_USE_RESPONSES_API` in `.env.example` |
| `127.0.0.1:2024` confusion | That's the LangGraph local API, not LangSmith. Don't set `LANGCHAIN_ENDPOINT` to localhost. |
