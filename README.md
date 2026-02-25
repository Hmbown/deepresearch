# Deep Research

A LangGraph multi-agent system that takes a research question, clarifies it if needed, fans out parallel web researchers, and pulls it all together into a cited report.

<p align="center">
  <img src="docs/architecture.svg" alt="Architecture" width="100%"/>
</p>

## Getting started

```bash
git clone https://github.com/Hmbown/deepresearch.git
cd deepresearch
pip install -e .
cp .env.example .env   # add your OPENAI_API_KEY here
deepresearch "your research question"
```

You'll need **Python 3.11+** and an **OpenAI API key**. By default, search uses OpenAI's built-in `web_search` tool in the Responses API. Tavily/Exa are optional alternatives.

## How it works

The pipeline has three stages:

1. **scope_intake** — figures out what you're actually asking. If the question is vague, it asks for clarification before doing any research.
2. **research_supervisor** — breaks the question into parallel research tracks, dispatches researcher agents via LangGraph `Send`, waits at a barrier, then decides whether to iterate or move on.
3. **final_report** — takes all the collected evidence and notes and synthesizes a report with citations.

## Configuration

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Check that everything works:

```bash
deepresearch --preflight
```

If you have LangSmith set up, you can verify tracing too:

```bash
deepresearch --preflight my-langsmith-project
```

## Usage

### CLI

```bash
deepresearch "your research question"
```

### Without web search

If you don't have search API credits or just want to test the pipeline:

```bash
SEARCH_PROVIDER=none deepresearch "key differences between Python 3.11 and 3.12"
```

### Use Exa or Tavily instead of OpenAI web search

```bash
SEARCH_PROVIDER=exa deepresearch "your research question"
# or
SEARCH_PROVIDER=tavily deepresearch "your research question"
```

### LangGraph Studio

Spin up the dev server and open it in Studio:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 \
  langgraph dev --host 127.0.0.1 --port 2024 --no-browser --allow-blocking
```

Then go to: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

<details>
<summary>What it looks like in Studio</summary>

![LangGraph Studio](docs/langgraph-graph.png)

</details>

## Project structure

| Module | What it does |
|---|---|
| `graph.py` | Wires the three pipeline stages together |
| `state.py` | Shared state, Pydantic schemas, helpers |
| `intake.py` | Clarification + brief generation with Command routing |
| `supervisor_subgraph.py` | Parallel research dispatch and iteration loop |
| `researcher_subgraph.py` | Individual researcher agent with search/fetch/think tools |
| `report.py` | Report synthesis with retries and source citations |
| `nodes.py` | Tool implementations — search, fetch (SSRF-protected), think |
| `prompts.py` | All the prompt templates |
| `config.py` | Model and search provider config |
| `env.py` | Env bootstrap and preflight checks |
| `cli.py` | CLI with streaming progress display |
| `evals/` | LLM-as-judge evaluation framework |

## LangSmith tracing

Add these to your `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=deepresearch-local
```

## Troubleshooting

| Issue | What to do |
|---|---|
| `deepagents` not found | Run `pip install -e .` again |
| Search credit errors | Set `SEARCH_PROVIDER=none` in `.env` |
| LangSmith shows no data | Check your billing/quota on the free tier |
| OpenAI Responses API issues | Check `OPENAI_USE_RESPONSES_API` in `.env.example` |
| Confused about `127.0.0.1:2024` | That's the LangGraph local API, not LangSmith. Don't set `LANGCHAIN_ENDPOINT` to localhost. |
