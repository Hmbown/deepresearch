import io
from types import SimpleNamespace

from deepresearch import cli


def _researcher_chain_start_event(topic: str) -> dict[str, object]:
    return {
        "event": "on_chain_start",
        "name": "LangGraph",
        "metadata": {"checkpoint_ns": "researcher|1"},
        "data": {"input": {"messages": [SimpleNamespace(type="human", content=topic)]}},
    }


def test_collect_evidence_from_research_output_ignores_model_cited_urls_for_progress_counts():
    output = {
        "messages": [
            SimpleNamespace(
                type="ai",
                content=(
                    "Analysts reported that autonomous testing reduced sortie planning time by 30 percent [1].\n\n"
                    "Sources:\n"
                    "[1] https://example.com/report"
                ),
            )
        ]
    }

    evidence_records, model_cited_count = cli._collect_evidence_from_research_output(output)
    assert evidence_records == []
    assert model_cited_count == 1


def test_progress_display_finishes_only_for_top_level_langgraph_event():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)

    nested_event = {
        "event": "on_chain_end",
        "name": "LangGraph",
        "metadata": {"checkpoint_ns": "research_supervisor|nested"},
        "data": {"output": {"messages": []}},
    }
    top_level_event = {
        "event": "on_chain_end",
        "name": "LangGraph",
        "metadata": {"checkpoint_ns": ""},
        "data": {"output": {"messages": []}},
    }

    assert display.handle_event(nested_event) is None
    assert display.handle_event(top_level_event) == {"messages": []}
    assert display.handle_event(top_level_event) == {"messages": []}

    lines = [line for line in stream.getvalue().splitlines() if "Full pipeline finished in" in line]
    assert len(lines) == 1


def test_progress_display_summarizes_recursion_limit_without_raw_error_url():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)
    display.handle_event(_researcher_chain_start_event("Counter-UAS integration updates"))

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
            "event": "on_custom_event",
            "name": "supervisor_progress",
            "metadata": {"checkpoint_ns": "research_supervisor|supervisor_finalize"},
            "data": {
                "supervisor_iteration": 1,
                "requested_research_units": 1,
                "dispatched_research_units": 1,
                "skipped_research_units": 0,
                "remaining_iterations": 15,
                "max_concurrent_research_units": 6,
                "max_researcher_iterations": 16,
                "quality_gate_status": "none",
                "quality_gate_reason": "",
                "evidence_record_count": 0,
                "source_domain_count": 0,
                "source_domains": [],
                "research_units": [
                    {
                        "topic": "Counter-UAS integration updates",
                        "status": "failed",
                        "failure_reason": "Recursion limit of 41 reached",
                        "duration_seconds": 2.1,
                    }
                ],
            },
        }
    )

    rendered = stream.getvalue()
    assert "encountered a recursion limit (41 steps)" in rendered
    assert "GRAPH_RECURSION_LIMIT" not in rendered


def test_progress_display_research_summary_uses_extracted_evidence():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)
    display.handle_event(_researcher_chain_start_event("Autonomous maritime systems"))
    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "deep-researcher",
            "metadata": {"checkpoint_ns": "researcher|1"},
            "data": {
                "output": {
                    "messages": [
                        SimpleNamespace(
                            type="tool",
                            name="search_web",
                            tool_call_id="search-1",
                            content="URL: https://navy.example/briefing",
                        ),
                        SimpleNamespace(
                            type="ai",
                            content=(
                                "The latest fleet trials showed autonomous maritime drones extending mission "
                                "endurance by 18 percent [1].\n\n"
                                "Sources:\n"
                                "[1] https://navy.example/briefing"
                            ),
                        ),
                    ]
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "1 sources, 1 domains" in rendered
    assert "Total so far: 1 sources from 1 domains" in rendered


def test_progress_display_research_summary_surfaces_model_cited_urls_separately():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)
    display.handle_event(_researcher_chain_start_event("Autonomous maritime systems"))
    display.handle_event(
        {
            "event": "on_chain_end",
            "name": "deep-researcher",
            "metadata": {"checkpoint_ns": "researcher|1"},
            "data": {
                "output": {
                    "messages": [
                        SimpleNamespace(
                            type="ai",
                            content=("Autonomy note [1].\n\nSources:\n[1] https://navy.example/briefing"),
                        )
                    ]
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "0 sources, 0 domains, 1 model-cited URLs" in rendered
