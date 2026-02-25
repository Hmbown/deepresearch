import asyncio
import importlib

from langchain_core.messages import AIMessage, ToolMessage

from deepresearch import report
from deepresearch.researcher_subgraph import extract_research_from_messages


def _load_graph_module():
    graph = importlib.import_module("deepresearch.graph")
    return importlib.reload(graph)


def test_extract_research_from_messages_emits_typed_evidence_records():
    messages = [
        ToolMessage(
            content=(
                "[Source 1] Example A\nURL: https://example.com/source-a\nSummary: Baseline metric improved 12% [1]."
            ),
            name="search_web",
            tool_call_id="search-1",
        ),
        AIMessage(
            content=(
                "Executive Summary\n"
                "- Baseline metric improved 12% [1].\n\n"
                "Evidence Log\n"
                "- Independent replication is uncertain and needs follow-up validation.\n\n"
                "Sources:\n"
                "[1] https://example.com/source-a"
            )
        ),
    ]

    compressed, raw_notes, evidence_ledger = extract_research_from_messages({"messages": messages})

    assert compressed is not None
    assert raw_notes
    assert evidence_ledger
    assert any(record.source_urls for record in evidence_ledger)
    assert all(record.source_type in {"fetched", "model_cited"} for record in evidence_ledger)
    assert any(record.source_type == "fetched" for record in evidence_ledger)


def test_supervisor_finalize_always_accepts_research_complete(monkeypatch):
    """ResearchComplete is always accepted — no quality gate."""
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    monkeypatch.setattr(supervisor_subgraph, "get_max_researcher_iterations", lambda: 6)

    state = {
        "pending_complete_calls": [{"id": "complete-1"}],
        "pending_requested_research_units": 0,
        "pending_dispatched_research_units": 0,
        "pending_skipped_research_units": 0,
        "pending_remaining_iterations": 4,
        "research_unit_summaries": [],
        "research_unit_summaries_consumed": 0,
        "evidence_ledger": [],
        "research_iterations": 2,
    }

    result = asyncio.run(supervisor_subgraph.supervisor_finalize(state))
    assert result["research_iterations"] == 6
    assert any("ResearchComplete received" in msg.content for msg in result["supervisor_messages"])


def test_extract_evidence_records_extracts_unique_urls():
    """Evidence extraction returns one record per unique URL."""
    from deepresearch.researcher_subgraph import _extract_evidence_records

    text = (
        "Key Findings\n"
        "- Renewables grew 30% year-over-year [1].\n"
        "- Battery costs fell below $100/kWh [2].\n\n"
        "Sources:\n"
        "- [1] https://example.com/renewables-report\n"
        "[2]: https://example.org/battery-costs\n"
        "Also see https://example.com/renewables-report for details."
    )
    records = _extract_evidence_records(text)
    assert records
    urls_found = [url for r in records for url in r.source_urls]
    assert "https://example.com/renewables-report" in urls_found
    assert "https://example.org/battery-costs" in urls_found
    # Duplicate URL should not create a second record
    assert len([u for u in urls_found if u == "https://example.com/renewables-report"]) == 1


def test_extract_evidence_records_deduplicates_urls():
    """Duplicate URLs across the text produce only one record each."""
    from deepresearch.researcher_subgraph import _extract_evidence_records

    text = (
        "First mention: https://example.com/a\n"
        "Second mention: https://example.com/a\n"
        "Different URL: https://example.org/b\n"
    )
    records = _extract_evidence_records(text)
    assert len(records) == 2
    all_urls = [url for r in records for url in r.source_urls]
    assert "https://example.com/a" in all_urls
    assert "https://example.org/b" in all_urls


def test_extract_evidence_records_empty_text():
    """Empty input returns no records."""
    from deepresearch.researcher_subgraph import _extract_evidence_records

    assert _extract_evidence_records("") == []
    assert _extract_evidence_records("   ") == []


def test_extract_evidence_records_no_urls():
    """Text without URLs returns no records."""
    from deepresearch.researcher_subgraph import _extract_evidence_records

    records = _extract_evidence_records("Just some plain text without any links.")
    assert records == []


def test_final_report_generation_preserves_evidence_ledger_source_transparency():
    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": [],
                "raw_notes": [],
                "evidence_ledger": [
                    {
                        "source_urls": ["https://example.com/a"],
                    }
                ],
                "final_report": "Synthesis from typed evidence.",
            }
        )
    )

    assert "Sources:" in result["final_report"]
    assert "https://example.com/a" in result["final_report"]
    assert "No source URLs were available in collected notes." not in result["final_report"]
