# Deep Research

LangGraph multi-agent system that clarifies scope when needed, runs focused web research in parallel, and synthesizes a cited answer. One canonical runtime path: `scope_intake -> research_supervisor -> final_report_generation`.

## Prerequisites

- Python 3.11+
- OpenAI API key (`OPENAI_API_KEY`) — required
- Tavily API key (`TAVILY_API_KEY`) — optional, for web search (default provider; Exa also supported)

## Install

```bash
git clone https://github.com/Hmbown/deepresearch.git
cd deepresearch
pipx install --force --editable .
```

## Configure

```bash
cp .env.example .env
# Edit .env — set at minimum OPENAI_API_KEY
```

## Preflight

```bash
deepresearch --preflight
```

Pass a LangSmith project name to also verify tracing auth:

```bash
deepresearch --preflight my-langsmith-project
```

## Run (CLI)

```bash
deepresearch "your research question"
```

## Run (LangGraph dev server + Studio)

Start the local dev server:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 \
  langgraph dev --host 127.0.0.1 --port 2024 --no-browser --allow-blocking
```

Then open Studio via LangSmith at:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## No-search fallback (smoke test)

If Exa credits are exhausted or you want to validate the runtime path without web search:

```bash
SEARCH_PROVIDER=none deepresearch "summarize the key differences between Python 3.11 and 3.12"
```

This exercises intake, supervisor dispatch, and report synthesis using only model knowledge.

## LangSmith tracing

Set these in `.env` to enable tracing:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=deepresearch-local
```

## Common gotchas

- **`http://127.0.0.1:2024` is the LangGraph local API**, not LangSmith. Do NOT set `LANGCHAIN_ENDPOINT` to localhost — that breaks cloud tracing. Leave `LANGCHAIN_ENDPOINT` unset (defaults to `https://api.smith.langchain.com`).
- **LangSmith shows "No data" but auth is valid**: check your LangSmith billing/quota limits — rate-limited free tiers surface this way.
- **Search credit errors** (Exa `402`, Tavily quota): set `SEARCH_PROVIDER=none` in `.env` to keep working without web search.
- **`deepagents` not found**: re-run `pip install -e .` — it's listed as a project dependency.
- **OpenAI Responses API issues**: the default config uses the Responses API. If you hit errors, check `OPENAI_USE_RESPONSES_API` in `.env.example`.
