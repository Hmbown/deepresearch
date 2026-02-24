# ADR-001: Git History Remediation

## Status

Accepted (2026-02-24)

## Context

A repository quality audit (SHA-780 through SHA-786) identified two categories of
unwanted content in git history:

1. **AI co-authorship trailers** -- Two commits contained `Co-Authored-By: Claude Opus 4.6`
   lines in their commit messages. These are inaccurate attribution for a project
   where the human author directed and reviewed all changes.

2. **Machine-specific paths** -- A local macOS volume path (containing the
   host disk name and absolute mount point) appeared in
   `docs/external_ai_review_prompt.md` across multiple commits, leaking a
   machine-specific filesystem path into the repository.

## Decision

**Rewrite history** using `git filter-repo` to:

- Strip all `Co-Authored-By:` lines from every commit message.
- Replace the machine-specific absolute path with the relative project name
  in all tracked file content across all commits.

This is preferred over "accept as-is" because the repository is pre-publication and
has no external consumers. A clean history is more important than preserving commit
hashes for a project being submitted as a portfolio piece.

## Consequences

- All commit SHAs change. Any existing references to old SHAs (Linear issues, etc.)
  become stale and must be updated manually.
- The GitHub remote must be force-pushed or replaced with a fresh repository.
- Future contributors should not add AI co-authorship trailers to commits.
