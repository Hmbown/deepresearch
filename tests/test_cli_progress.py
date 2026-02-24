import io
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from deepresearch import cli


def test_collect_evidence_from_research_output_uses_message_extraction():
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

    evidence_records, source_urls = cli._collect_evidence_from_research_output(output)
    assert len(evidence_records) >= 1
    assert any("https://example.com/report" in record.source_urls for record in evidence_records)
    assert "https://example.com/report" in source_urls


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
    display._topic_queue.append("Counter-UAS integration updates")

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "LangGraph",
            "metadata": {"checkpoint_ns": "researcher|1"},
            "data": {},
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
                        ToolMessage(
                            content=(
                                "[Research unit failed: Recursion limit of 41 reached without hitting a stop "
                                "condition. For troubleshooting, visit: "
                                "https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT]"
                            ),
                            name="ConductResearch",
                            tool_call_id="call-1",
                        )
                    ]
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "hit tool call limit (41 steps)" in rendered
    assert "GRAPH_RECURSION_LIMIT" not in rendered


def test_progress_display_research_summary_uses_extracted_evidence():
    stream = io.StringIO()
    display = cli.ProgressDisplay(stream=stream)
    display._topic_queue.append("Autonomous maritime systems")

    display.handle_event(
        {
            "event": "on_chain_start",
            "name": "LangGraph",
            "metadata": {"checkpoint_ns": "researcher|1"},
            "data": {},
        }
    )
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
                            content=(
                                "The latest fleet trials showed autonomous maritime drones extending mission "
                                "endurance by 18 percent [1].\n\n"
                                "Sources:\n"
                                "[1] https://navy.example/briefing"
                            ),
                        )
                    ]
                }
            },
        }
    )

    rendered = stream.getvalue()
    assert "1 evidence records, 1 domains" in rendered
    assert "Total so far: 1 evidence records from 1 domains" in rendered
