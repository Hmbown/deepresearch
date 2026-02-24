from __future__ import annotations

import io

from deepresearch import cli


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


def test_progress_display_quality_gate_rejection_is_user_visible():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {
                "output": {
                    "runtime_progress": {
                        "supervisor_iteration": 2,
                        "requested_research_units": 0,
                        "dispatched_research_units": 0,
                        "skipped_research_units": 0,
                        "remaining_iterations": 10,
                        "max_concurrent_research_units": 6,
                        "max_researcher_iterations": 16,
                        "quality_gate_status": "retry",
                        "quality_gate_reason": "insufficient_source_domains",
                        "evidence_record_count": 1,
                        "source_domain_count": 1,
                        "source_domains": ["example.com"],
                        "research_units": [],
                    }
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "Quality gate:" in rendered
    assert "RETRY" in rendered
    assert "insufficient_source_domains" in rendered


def test_progress_display_wave_dispatch_uses_runtime_progress_payload():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {"input": {}},
        }
    )

    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {
                "output": {
                    "runtime_progress": {
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
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "dispatching 2 researchers in parallel" in rendered


def test_progress_display_ignores_supervisor_tool_message_text_when_runtime_progress_present():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {"input": {}},
        }
    )
    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {
                "output": {
                    "supervisor_messages": [
                        {
                            "type": "tool",
                            "content": "[ResearchComplete received]",
                            "name": "ResearchComplete",
                        }
                    ],
                    "runtime_progress": {
                        "supervisor_iteration": 3,
                        "requested_research_units": 0,
                        "dispatched_research_units": 0,
                        "skipped_research_units": 0,
                        "remaining_iterations": 7,
                        "max_concurrent_research_units": 6,
                        "max_researcher_iterations": 16,
                        "quality_gate_status": "retry",
                        "quality_gate_reason": "insufficient_evidence_records",
                        "evidence_record_count": 1,
                        "source_domain_count": 1,
                        "source_domains": ["example.com"],
                        "research_units": [],
                    },
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "RETRY" in rendered
    assert "insufficient_evidence_records" in rendered
    assert "PASS" not in rendered


def test_progress_display_no_dispatch_or_gate_output_when_runtime_progress_absent():
    """Parity guard: CLI must NOT render dispatch/gate info from supervisor_messages alone."""
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {"input": {}},
        }
    )
    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
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

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {"input": {}},
        }
    )
    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {
                "output": {
                    "runtime_progress": {
                        "supervisor_iteration": 1,
                        "dispatched_research_units": 3,
                    }
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "dispatching 3 researchers in parallel" in rendered
    assert "Quality gate" not in rendered


def test_progress_display_does_not_derive_dispatch_count_from_tool_call_count():
    """Parity guard: dispatch count comes from runtime_progress, not from counting tool calls."""
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {"input": {}},
        }
    )
    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "supervisor_tools",
            "metadata": {"checkpoint_ns": "research_supervisor|tools"},
            "data": {
                "output": {
                    "supervisor_messages": [
                        {"type": "tool", "content": "Topic A findings", "name": "ConductResearch"},
                        {"type": "tool", "content": "Topic B findings", "name": "ConductResearch"},
                        {"type": "tool", "content": "Topic C findings", "name": "ConductResearch"},
                    ],
                    "runtime_progress": {
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
                    },
                }
            },
        }
    )

    rendered = stream.getvalue()
    # CLI must show 2 (from runtime_progress), not 3 (from tool message count)
    assert "dispatching 2 researchers in parallel" in rendered
    assert "1 deferred by runtime caps" in rendered
