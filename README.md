# deepresearch

Open-source LangGraph deep research agent with one runtime path:

`route_turn -> clarify_with_user -> write_research_brief -> research_supervisor -> final_report_generation`

## Quickstart

```bash
git clone https://github.com/Hmbown/deepresearch.git
cd deepresearch
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
[ -f .env ] || cp .env.example .env
python -m deepresearch.cli --preflight
python -m deepresearch.cli "Compare retrieval strategies for production RAG systems"
```

## Multi-turn Usage

Yes, it supports more than 2 turns.

- Reuse the same `thread_id`
- Pass previous `messages` into each next `run(...)`

```python
import asyncio
from deepresearch.cli import run

async def main():
    thread_id = "demo-thread"
    messages = []
    for turn in [
        "Please research this topic for me.",
        "Scope to U.S. and EU from 2020-2026.",
        "Now focus on policy blockers only.",
    ]:
        result = await run(turn, thread_id=thread_id, prior_messages=messages)
        messages = list(result.get("messages", messages))
        print(result.get("intake_decision"))

asyncio.run(main())
```

## Minimal Config

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Yes | — |
| `EXA_API_KEY` | No | — |
| `ORCHESTRATOR_MODEL` | No | `openai:gpt-5.2` |
| `SUBAGENT_MODEL` | No | `openai:gpt-5.2` |
| `MAX_CONCURRENT_RESEARCH_UNITS` | No | `4` |
| `MAX_RESEARCHER_ITERATIONS` | No | `6` |
| `LANGCHAIN_TRACING_V2` | No | `false` |
| `LANGCHAIN_API_KEY` | No | — |
| `LANGCHAIN_PROJECT` | No | `deepresearch-local` |

See `.env.example` for full options.

## Validation

```bash
python -m deepresearch.cli --preflight
python -m compileall src/deepresearch
python -m pytest -q
python scripts/validate_multiagent_runtime.py
```

Expected long validation outcomes:

- `turn1.decision=clarify`
- `turn2.decision=proceed`
- `turn2.conduct_research_count>=3`

## LangSmith Placeholder

![LangSmith Studio setup placeholder](docs/images/langsmith-studio-setup-placeholder.png)

## Open Source

- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [License (MIT)](LICENSE)
