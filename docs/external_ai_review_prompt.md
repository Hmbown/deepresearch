You are an independent senior reviewer auditing current runtime hardening work in this repository.

Repository: `deepresearch`
Primary goal: validate SHA-765 through SHA-768 end-to-end.

Context:
- Researcher execution is currently deep-agent-backed (`create_deep_agent` in `src/deepresearch/researcher_subgraph.py`).
- The canonical runtime path remains `route_turn -> clarify_with_user -> write_research_brief -> research_supervisor -> final_report_generation`.
- Online eval tooling exists in `src/deepresearch/evals/*` and `scripts/run_online_evals.py`.
- Contributor guidance should match implemented runtime behavior and deferred scope.

Review mode:
- Use a strict code-review mindset.
- Prioritize bugs, regressions, missing acceptance criteria, unsafe assumptions, and test gaps.
- Findings first, ordered by severity, with file references and line numbers.
- Keep summaries brief and secondary.

Required files to inspect:
- `pyproject.toml`
- `.github/workflows/test.yml`
- `.github/workflows/online-evals.yml`
- `src/deepresearch/researcher_subgraph.py`
- `src/deepresearch/evals/evaluators.py`
- `scripts/run_online_evals.py`
- `tests/test_evals.py`
- `tests/test_graph_native_subgraphs.py`
- `tests/test_researcher_subgraph_integration.py`
- `agents.md`

Acceptance criteria to validate:
1. SHA-765 (restore Ruff gate)
- `ruff check .` is green.
- The known unused-import violations are removed without behavior changes.

2. SHA-766 (resolve yanked deepagents pin)
- `pyproject.toml` no longer pins a yanked deepagents version.
- Clean install path has no yanked warning for selected deepagents pin.
- Tests stay green after the dependency decision.

3. SHA-767 (operationalize online eval harness)
- A standard automation entrypoint exists via GitHub Actions manual dispatch.
- Workflow accepts project/since/limit inputs.
- Workflow logs include a usable score summary and handles no-runs/no-traces gracefully.

4. SHA-768 (reconcile stale deferred docs/issues)
- Repository guidance reflects current deep-agent + online-eval implementation.
- Deferred items for SHA-752/SHA-753 reflect current scope (implemented vs remaining work).
- No contradictory architecture claims remain in contributor guidance.

Commands to run:
1. `.venv/bin/python -m ruff check .`
2. `.venv/bin/python -m pytest -q`
3. `.venv/bin/python -m pip index versions deepagents`
4. `rg -n "deepagents==|create_deep_agent|online_evals|run_online_evals|workflow_dispatch|schedule:" pyproject.toml src scripts .github/workflows tests agents.md`
5. `.venv/bin/python scripts/run_online_evals.py --help`

Output format (mandatory):
1. Findings
- Use severity labels: `Critical`, `High`, `Medium`, `Low`.
- Include `file:line` references and concrete impact.

2. Open questions / assumptions
- Only items that materially affect correctness.

3. Acceptance matrix
- One line each for SHA-765..SHA-768: `Pass`, `Partial`, or `Fail` with one-sentence justification.

4. Residual risk
- Briefly call out what could still fail in production even if tests pass.
