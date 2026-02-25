from __future__ import annotations

import io

from deepresearch import cli


def _supervisor_prepare_start_event() -> dict[str, object]:
    return {
        "event": "on_chain_start",
        "name": "supervisor_prepare",
        "metadata": {"checkpoint_ns": "research_supervisor|supervisor_prepare"},
        "data": {"input": {}},
    }


def _supervisor_progress_event(payload: dict[str, object]) -> dict[str, object]:
    return {
        "event": "on_custom_event",
        "name": "supervisor_progress",
        "metadata": {"checkpoint_ns": "research_supervisor|supervisor_finalize"},
        "data": payload,
    }


def test_progress_display_search_summary_renders_count_and_domains():
    stream = io.StringIO()
    display = cli.ProgressDisplay(verbose=True, stream=stream)

    display.handle_event(
        {
            "event": "on_tool_end",
            "name": "search_web",
            "metadata": {"checkpoint_ns": "researcher|1"},
            "data": {
                "output": (
                    "[Source 1] Gemini deep research\n"
                    "URL: https://blog.google/ai/research\n"
                    "[Source 2] OpenAI\n"
                    "URL: https://openai.com/research\n"
                )
            },
        }
    )

    rendered = stream.getvalue()
    assert "-> 2 results (blog.google, openai.com)" in rendered


def test_progress_display_search_summary_handles_no_results():
    stream = io.StringIO()
    display = cli.ProgressDisplay(verbose=True, stream=stream)

    display.handle_event(
        {
            "event": "on_tool_end",
            "name": "search_web",
            "metadata": {"checkpoint_ns": "researcher|1"},
            "data": {"output": "No relevant search results found."},
        }
    )

    rendered = stream.getvalue()
    assert "-> 0 results" in rendered


def test_progress_display_search_summary_truncates_domain_list():
    stream = io.StringIO()
    display = cli.ProgressDisplay(verbose=True, stream=stream)

    display.handle_event(
        {
            "event": "on_tool_end",
            "name": "search_web",
            "metadata": {"checkpoint_ns": "researcher|1"},
            "data": {
                "output": (
                    "[Source 1] A\nURL: https://a.com/x\n"
                    "[Source 2] B\nURL: https://b.org/y\n"
                    "[Source 3] C\nURL: https://c.net/z\n"
                    "[Source 4] D\nURL: https://d.io/q\n"
                )
            },
        }
    )

    rendered = stream.getvalue()
    assert "a.com, b.org, c.net, ..." in rendered


def test_progress_display_wave_summary_shows_source_counts():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())
    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 2,
                "requested_research_units": 1,
                "dispatched_research_units": 1,
                "skipped_research_units": 0,
                "remaining_iterations": 10,
                "max_concurrent_research_units": 6,
                "max_researcher_iterations": 16,
                "evidence_record_count": 5,
                "source_domain_count": 3,
                "source_domains": ["example.com", "example.org", "example.net"],
                "research_units": [],
            }
        )
    )

    rendered = stream.getvalue()
    assert "5 sources" in rendered
    assert "3 domains" in rendered


def test_progress_display_does_not_render_empty_wave_summary_when_no_dispatch():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())
    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 2,
                "requested_research_units": 0,
                "dispatched_research_units": 0,
                "skipped_research_units": 0,
                "remaining_iterations": 10,
                "max_concurrent_research_units": 6,
                "max_researcher_iterations": 16,
                "evidence_record_count": 5,
                "source_domain_count": 3,
                "source_domains": ["example.com", "example.org", "example.net"],
                "research_units": [],
            }
        )
    )

    rendered = stream.getvalue()
    assert "evaluating quality gate" in rendered
    assert "Wave 1 complete" not in rendered


def test_progress_display_wave_dispatch_uses_runtime_progress_payload():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())

    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 1,
                "requested_research_units": 2,
                "dispatched_research_units": 2,
                "skipped_research_units": 0,
                "remaining_iterations": 16,
                "max_concurrent_research_units": 6,
                "max_researcher_iterations": 16,
                "quality_gate_status": "none",
                "quality_gate_reason": "",
                "evidence_record_count": 0,
                "source_domain_count": 0,
                "source_domains": [],
                "research_units": [],
            }
        )
    )

    rendered = stream.getvalue()
    assert "dispatching 2 researchers in parallel" in rendered


def test_progress_display_renders_planned_research_tracks_for_wave():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())
    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 1,
                "requested_research_units": 2,
                "dispatched_research_units": 2,
                "skipped_research_units": 0,
                "remaining_iterations": 16,
                "max_concurrent_research_units": 6,
                "max_researcher_iterations": 16,
                "quality_gate_status": "none",
                "quality_gate_reason": "",
                "evidence_record_count": 0,
                "source_domain_count": 0,
                "source_domains": [],
                "planned_research_units": [
                    {
                        "call_id": "call-1",
                        "topic": "Identify going-concern and filing-based distress signals.",
                    },
                    {
                        "call_id": "call-2",
                        "topic": "Collect market and credit distress indicators for U.S. issuers.",
                    },
                ],
                "research_units": [],
            }
        )
    )

    rendered = stream.getvalue()
    assert "Research tracks:" in rendered
    assert "[1] Identify going-concern and filing-based distress signals." in rendered
    assert "[2] Collect market and credit distress indicators for U.S. issuers." in rendered


def test_progress_display_failed_track_uses_planned_track_index_label():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())
    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 1,
                "requested_research_units": 2,
                "dispatched_research_units": 2,
                "skipped_research_units": 0,
                "remaining_iterations": 16,
                "max_concurrent_research_units": 6,
                "max_researcher_iterations": 16,
                "quality_gate_status": "none",
                "quality_gate_reason": "",
                "evidence_record_count": 0,
                "source_domain_count": 0,
                "source_domains": [],
                "planned_research_units": [
                    {"call_id": "call-1", "topic": "Track one topic"},
                    {"call_id": "call-2", "topic": "Track two topic"},
                ],
                "research_units": [
                    {
                        "call_id": "call-2",
                        "topic": "Track two topic",
                        "status": "failed",
                        "failure_reason": "Recursion limit of 41 reached",
                        "duration_seconds": 2.0,
                    }
                ],
            }
        )
    )

    rendered = stream.getvalue()
    assert 'track[2] "Track two topic" encountered a recursion limit (41 steps)' in rendered


def test_progress_display_uses_runtime_progress_not_tool_messages():
    """Verify CLI reads source counts from runtime_progress payload, not from tool messages."""
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())
    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 3,
                "requested_research_units": 1,
                "dispatched_research_units": 1,
                "skipped_research_units": 0,
                "remaining_iterations": 7,
                "max_concurrent_research_units": 6,
                "max_researcher_iterations": 16,
                "evidence_record_count": 1,
                "source_domain_count": 1,
                "source_domains": ["example.com"],
                "research_units": [],
            }
        )
    )

    rendered = stream.getvalue()
    assert "1 sources" in rendered
    assert "1 domains" in rendered


def test_progress_display_no_dispatch_or_gate_output_when_runtime_progress_absent():
    """Parity guard: CLI must NOT render dispatch/gate info from supervisor_messages alone."""
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "supervisor_prepare",
            "metadata": {"checkpoint_ns": "research_supervisor|supervisor_prepare"},
            "data": {"input": {}},
        }
    )
    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "supervisor_finalize",
            "metadata": {"checkpoint_ns": "research_supervisor|supervisor_finalize"},
            "data": {
                "output": {
                    "supervisor_messages": [
                        {
                            "type": "tool",
                            "content": "[ResearchComplete received]",
                            "name": "ResearchComplete",
                        },
                        {
                            "type": "tool",
                            "content": "Research findings on topic A",
                            "name": "ConductResearch",
                        },
                    ],
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "dispatching" not in rendered
    assert "Quality gate" not in rendered
    assert "PASS" not in rendered
    assert "RETRY" not in rendered


def test_progress_display_handles_partial_runtime_progress_gracefully():
    """Parity guard: partial runtime_progress must not crash or produce garbage."""
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())
    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 1,
                "dispatched_research_units": 3,
            }
        )
    )

    rendered = stream.getvalue()
    assert "dispatching 3 researchers in parallel" in rendered
    assert "Quality gate" not in rendered


def test_progress_display_does_not_derive_dispatch_count_from_tool_call_count():
    """Parity guard: dispatch count comes from runtime_progress, not from counting tool calls."""
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(_supervisor_prepare_start_event())
    display.handle_event(
        _supervisor_progress_event(
            {
                "supervisor_iteration": 1,
                "requested_research_units": 3,
                "dispatched_research_units": 2,
                "skipped_research_units": 1,
                "remaining_iterations": 14,
                "max_concurrent_research_units": 2,
                "max_researcher_iterations": 16,
                "quality_gate_status": "none",
                "quality_gate_reason": None,
                "evidence_record_count": 0,
                "source_domain_count": 0,
                "source_domains": [],
                "research_units": [],
            }
        )
    )

    rendered = stream.getvalue()
    # CLI must show 2 (from runtime_progress), not 3 (from tool message count)
    assert "dispatching 2 researchers in parallel" in rendered
    assert "1 deferred by runtime caps" in rendered
