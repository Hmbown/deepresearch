# Linear Follow-Ups (2026-02-24)

Created from repository quality audit findings.

## Issues

- SHA-780: Add pytest to dev extras so documented test setup works
  - https://linear.app/shannon-labs/issue/SHA-780/add-pytest-to-dev-extras-so-documented-test-setup-works
- SHA-781: Align CONTRIBUTING setup with required developer commands
  - https://linear.app/shannon-labs/issue/SHA-781/align-contributing-setup-with-required-developer-commands
- SHA-782: Document missing env vars in README config section
  - https://linear.app/shannon-labs/issue/SHA-782/document-missing-env-vars-in-readme-config-section
- SHA-783: Make README project structure listing reflect current src/deepresearch layout
  - https://linear.app/shannon-labs/issue/SHA-783/make-readme-project-structure-listing-reflect-current-srcdeepresearch
- SHA-784: Update stale runtime path in docs/external_ai_review_prompt.md
  - https://linear.app/shannon-labs/issue/SHA-784/update-stale-runtime-path-in-docsexternal-ai-review-promptmd
- SHA-785: Add .benchmarks and .deepeval to .gitignore
  - https://linear.app/shannon-labs/issue/SHA-785/add-benchmarks-and-deepeval-to-gitignore
- SHA-786: Decide and document remediation for machine-specific path present in git history
  - https://linear.app/shannon-labs/issue/SHA-786/decide-and-document-remediation-for-machine-specific-path-present-in

## Short Prompt For Next AI

Complete SHA-780 through SHA-786 with production-quality fixes. Implement code/doc updates for SHA-780..SHA-785, run `ruff check .` and `pytest -q`, and update each Linear issue with what changed and verification output. For SHA-786, propose a concrete policy decision (rewrite history vs accept as-is), document the decision in-repo, and link that commit back to the issue. Open a single focused PR summarizing all fixes and residual risk.
