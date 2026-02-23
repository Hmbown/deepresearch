You are an independent senior reviewer auditing a major runtime cutover in this repository.

Repository: `deepresearch`
Primary goal: validate SHA-758 through SHA-764 end-to-end.

Context:
- This codebase replaced a `deepagents` runtime with a native LangGraph multi-agent architecture.
- Cutover scope includes state schemas, prompts, researcher/supervisor subgraphs, main-graph wiring, CLI behavior, and runtime verification.
- This project currently prioritizes a single canonical runtime path (no dual pipelines).
- Current expected runtime defaults:
  - `ORCHESTRATOR_MODEL=openai:gpt-5.2`
  - `SUBAGENT_MODEL=openai:gpt-5.2`
  - Exa search provider in runtime.

Review mode:
- Use a strict code-review mindset.
- Prioritize bugs, regressions, missing acceptance criteria, unsafe assumptions, and test gaps.
- Findings first, ordered by severity, with file references and line numbers.
- Keep summaries brief and secondary.

Required files to inspect:
- `src/deepresearch/graph.py`
- `src/deepresearch/prompts.py`
- `src/deepresearch/config.py`
- `src/deepresearch/cli.py`
- `pyproject.toml`
- `langgraph.json`
- `tests/test_graph_native_subgraphs.py`
- `tests/test_intake_scope_phase.py`
- `tests/test_prompts_contract.py`
- `tests/test_config_and_cli.py`
- `tests/test_architecture_guardrails.py`

Acceptance criteria to validate:
1. SHA-758 (deepagents removal + native LangGraph)
- Zero runtime references to `deepagents`, `create_deep_agent`, `FilesystemBackend`.
- Main runtime uses native subgraph composition.
- Single canonical path preserved.

2. SHA-759 (state schemas + structured outputs)
- `ConductResearch` and `ResearchComplete` models exist and are used as supervisor tools.
- Researcher/supervisor/main states align with intended reducers and state flow.

3. SHA-760 (prompts)
- Prompts are native to new architecture (no `write_todos`, `task()`, deepagents concepts).
- Includes supervisor, researcher, compression, and final-report prompts.
- Clarify/brief prompts remain intact.

4. SHA-761 (researcher subgraph)
- Researcher loop exists and compiles.
- Tool calls execute in parallel (`asyncio.gather`).
- Iteration cap enforced.
- Compression step returns compressed + raw notes.

5. SHA-762 (supervisor subgraph)
- Supervisor delegates via structured tool calls.
- Parallel researcher dispatch works and concurrency caps are enforced.
- Iteration cap and exit conditions are implemented.

6. SHA-763 (main graph wiring + cleanup)
- Main graph route is: `write_research_brief -> research_supervisor -> final_report_generation -> END`.
- No legacy manager/deepagents node path remains.
- CLI prefers `final_report` fallbacking to last AI message.
- `deepagents` dependency removed from `pyproject.toml`.

7. SHA-764 (verification)
- Tests pass.
- CLI clarification and response flows work.
- `langgraph dev` starts for Studio.
- Note any environment blockers (for example LangSmith auth).

Commands to run:
1. `.venv/bin/python -m pytest -q`
2. `rg -n "deepagents|create_deep_agent|FilesystemBackend|research_manager_node|build_research_manager|_get_research_manager" src tests pyproject.toml`
3. `.venv/bin/python -m deepresearch.cli "Compare the best options"`
4. `MAX_RESEARCHER_ITERATIONS=1 MAX_CONCURRENT_RESEARCH_UNITS=1 MAX_REACT_TOOL_CALLS=1 RESEARCHER_SIMPLE_SEARCH_BUDGET=1 RESEARCHER_COMPLEX_SEARCH_BUDGET=1 .venv/bin/python -m deepresearch.cli "What is retrieval-augmented generation?"`
5. `zsh -lc '.venv/bin/langgraph dev --no-browser --port 2030 >/tmp/langgraph_dev_2030.log 2>&1 & pid=$!; sleep 8; curl -sS --max-time 5 http://127.0.0.1:2030/docs | head -n 3; kill $pid >/dev/null 2>&1 || true; wait $pid >/dev/null 2>&1 || true'`
6. Optional (if LangSmith key is valid): list recent runs for project `deepresearch`.

Output format (mandatory):
1. Findings
- Use severity labels: `Critical`, `High`, `Medium`, `Low`.
- Include `file:line` references and concrete impact.

2. Open questions / assumptions
- Only items that materially affect correctness.

3. Acceptance matrix
- One line each for SHA-758..SHA-764: `Pass`, `Partial`, or `Fail` with one-sentence justification.

4. Residual risk
- Briefly call out what could still fail in production even if tests pass.
