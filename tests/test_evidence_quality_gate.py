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
                "[Source 1] Example A\n"
                "URL: https://example.com/source-a\n"
                "Summary: Baseline metric improved 12% [1]."
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
    assert all(0.0 <= record.confidence <= 1.0 for record in evidence_ledger)
    assert any(record.contradiction_or_uncertainty for record in evidence_ledger)


def test_supervisor_quality_gate_rejects_research_complete_without_evidence(monkeypatch):
    graph = _load_graph_module()
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    monkeypatch.setattr(supervisor_subgraph, "get_max_researcher_iterations", lambda: 6)

    state = {
        "supervisor_messages": [
            AIMessage(content="", tool_calls=[{"id": "complete-1", "name": "ResearchComplete", "args": {}}]),
        ],
        "notes": [],
        "raw_notes": [],
        "evidence_ledger": [],
        "research_iterations": 2,
    }

    result = asyncio.run(graph.supervisor_tools(state))
    assert result["research_iterations"] == 3
    assert any("ResearchComplete rejected" in msg.content for msg in result["supervisor_messages"])
    assert result["runtime_progress"]["quality_gate_status"] == "retry"
    assert result["runtime_progress"]["quality_gate_reason"] == "insufficient_evidence_records"


def test_supervisor_quality_gate_accepts_research_complete_with_sufficient_evidence(monkeypatch):
    graph = _load_graph_module()
    supervisor_subgraph = importlib.import_module("deepresearch.supervisor_subgraph")
    monkeypatch.setattr(supervisor_subgraph, "get_max_researcher_iterations", lambda: 6)

    state = {
        "supervisor_messages": [
            AIMessage(content="", tool_calls=[{"id": "complete-1", "name": "ResearchComplete", "args": {}}]),
        ],
        "notes": [],
        "raw_notes": [],
        "evidence_ledger": [
            {
                "claim": "Claim one [1].",
                "source_urls": ["https://example.com/a"],
                "confidence": 0.8,
                "contradiction_or_uncertainty": None,
            },
            {
                "claim": "Claim two [2].",
                "source_urls": ["https://example.org/b"],
                "confidence": 0.7,
                "contradiction_or_uncertainty": "Evidence is mixed across regions.",
            },
        ],
        "research_iterations": 2,
    }

    result = asyncio.run(graph.supervisor_tools(state))
    assert result["research_iterations"] == 6
    assert any("ResearchComplete received" in msg.content for msg in result["supervisor_messages"])
    assert result["runtime_progress"]["quality_gate_status"] == "pass"
    assert result["runtime_progress"]["source_domain_count"] == 2


def test_extract_evidence_records_handles_dash_prefixed_citation_urls():
    """Verify citation map parses '- [N] URL' and '[N]: URL' formats."""
    from deepresearch.researcher_subgraph import _extract_evidence_records

    text = (
        "Key Findings\n"
        "- Renewables grew 30% year-over-year [1].\n"
        "- Battery costs fell below $100/kWh [2].\n\n"
        "Sources:\n"
        "- [1] https://example.com/renewables-report\n"
        "[2]: https://example.org/battery-costs"
    )
    records = _extract_evidence_records(text)
    assert records
    urls_found = [url for r in records for url in r.source_urls]
    assert "https://example.com/renewables-report" in urls_found
    assert "https://example.org/battery-costs" in urls_found


def test_extract_evidence_records_multi_url_fallback_links_cited_claims():
    """Claims with citation markers get global URLs when citation map has gaps."""
    from deepresearch.researcher_subgraph import _extract_evidence_records

    text = (
        "Executive Summary\n"
        "The market expanded significantly in 2025 [1].\n"
        "Competition intensified among top players [2].\n\n"
        "Evidence Log\n"
        "- Market share data confirms growth trajectory [1].\n"
        "- Strategic pivots were noted across the sector [2].\n\n"
        "https://example.com/market-data\n"
        "https://example.org/strategy-report\n"
    )
    records = _extract_evidence_records(text)
    assert records
    cited_records = [r for r in records if r.source_urls]
    assert len(cited_records) >= 1, "Claims with citation markers should get fallback URLs"


def test_extract_evidence_records_source_section_bare_url_lines_excluded():
    """Source section lines that are bare URLs should not leak into claim lines."""
    from deepresearch.researcher_subgraph import _extract_evidence_records

    text = (
        "Key Findings\n"
        "- Important finding with evidence [1].\n\n"
        "Sources:\n"
        "- https://example.com/source-a\n"
        "- https://example.org/source-b\n"
    )
    records = _extract_evidence_records(text)
    # Source URL lines should not appear as claim records
    claim_texts = [r.claim for r in records]
    assert not any("https://example.com/source-a" == c for c in claim_texts)
    assert not any("https://example.org/source-b" == c for c in claim_texts)


def test_final_report_generation_preserves_evidence_ledger_source_transparency():
    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": [],
                "raw_notes": [],
                "evidence_ledger": [
                    {
                        "claim": "Claim one.",
                        "source_urls": ["https://example.com/a"],
                        "confidence": 0.8,
                        "contradiction_or_uncertainty": None,
                    }
                ],
                "final_report": "Synthesis from typed evidence.",
            }
        )
    )

    assert "Sources:" in result["final_report"]
    assert "https://example.com/a" in result["final_report"]
    assert "No source URLs were available in collected notes." not in result["final_report"]
