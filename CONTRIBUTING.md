# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
[ -f .env ] || cp .env.example .env
python -m deepresearch.cli --preflight
```

## Before Opening a PR

```bash
python -m compileall src/deepresearch
python -m pytest -q
```

## Architecture Rules

- Keep one canonical runtime path.
- No dual pipelines or transition flags.
- Keep behavior simple: clarify, delegate focused research, synthesize final answer.

## Pull Requests

- Keep PRs focused and small.
- Include tests for behavior changes.
- Update README when user-facing behavior changes.
