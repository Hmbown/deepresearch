"""Batch script to run online evaluations on recent LangSmith traces.

Usage:
    python scripts/run_online_evals.py --project deepresearch-local --since 24h --limit 50
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
import re

from deepresearch.env import bootstrap_env


def _parse_since(since: str) -> datetime:
    """Parse a relative time string like '24h', '7d', '1w' into a UTC datetime."""
    since = since.strip().lower()
    match = re.fullmatch(r"(\d+)([hdw])", since)
    if not match:
        raise ValueError(f"Invalid --since format: {since!r}. Use e.g. '24h', '7d', '1w'.")

    value = int(match.group(1))
    if value <= 0:
        raise ValueError("The --since value must be a positive integer.")

    now = datetime.now(timezone.utc)
    unit = match.group(2)
    if unit == "h":
        return now - timedelta(hours=value)
    if unit == "d":
        return now - timedelta(days=value)
    return now - timedelta(weeks=value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run online evaluations on recent LangSmith traces.")
    parser.add_argument("--project", default="deepresearch-local", help="LangSmith project name.")
    parser.add_argument("--since", default="24h", help="Time window (e.g. 24h, 7d, 1w).")
    parser.add_argument("--limit", type=int, default=50, help="Max number of runs to evaluate.")
    args = parser.parse_args()

    bootstrap_env()

    from langsmith import Client

    from deepresearch.evals.evaluators import eval_composite

    client = Client()
    try:
        start_time = _parse_since(args.since)
    except ValueError as exc:
        print(f"Invalid time window: {exc}")
        return 2

    print(f"Fetching root runs from project '{args.project}' since {start_time.isoformat()}...")
    runs = list(
        client.list_runs(
            project_name=args.project,
            is_root=True,
            start_time=start_time,
            limit=args.limit,
        )
    )

    if not runs:
        print("No runs found.")
        return 0

    print(f"Found {len(runs)} run(s). Checking for existing scores...\n")

    scored = 0
    skipped = 0
    results_table: list[dict] = []

    for run in runs:
        run_id = str(run.id)

        try:
            feedbacks = list(client.list_feedback(run_ids=[run_id]))
            existing_keys = {fb.key for fb in feedbacks}
        except Exception:
            existing_keys = set()

        if "composite_quality" in existing_keys:
            skipped += 1
            continue

        print(f"Scoring run {run_id[:12]}... ", end="", flush=True)
        try:
            result = eval_composite(run, client)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue

        answer_score = result.get("answer_result", {}).get("score")
        process_score = result.get("process_result", {}).get("score")
        composite_score = result.get("score")

        for feedback in [result.get("answer_result", {}), result.get("process_result", {}), result]:
            key = feedback.get("key")
            score = feedback.get("score")
            comment = feedback.get("comment", "")
            if key and score is not None:
                try:
                    client.create_feedback(run_id=run_id, key=key, score=score, comment=comment[:4000])
                except Exception:
                    pass

        scored += 1
        results_table.append({
            "run_id": run_id[:12],
            "answer": answer_score,
            "process": process_score,
            "composite": composite_score,
        })
        print(f"answer={answer_score} process={process_score} composite={composite_score}")

    print(f"\n{'='*60}")
    print(f"Scored: {scored} | Skipped (already scored): {skipped} | Total: {len(runs)}")

    if results_table:
        print(f"\n{'Run ID':<14} {'Answer':>8} {'Process':>9} {'Composite':>10}")
        print("-" * 45)
        for row in results_table:
            a = f"{row['answer']:.3f}" if row["answer"] is not None else "N/A"
            p = f"{row['process']:.3f}" if row["process"] is not None else "N/A"
            c = f"{row['composite']:.4f}" if row["composite"] is not None else "N/A"
            print(f"{row['run_id']:<14} {a:>8} {p:>9} {c:>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
