import asyncio

from langchain_core.messages import AIMessage

from deepresearch import report
from deepresearch.state import FALLBACK_FINAL_REPORT


class _FakeReportModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages, config=None):
        del config
        self.calls.append(messages)
        if self._responses:
            response = self._responses.pop(0)
        else:
            response = AIMessage(content="")
        if isinstance(response, BaseException):
            raise response
        return response


def test_final_report_generation_success_path_preserves_source_markers(monkeypatch):
    model = _FakeReportModel(
        [
            AIMessage(
                content=(
                    "## Summary\n"
                    "Primary finding with evidence [1].\n\n"
                    "Sources:\n"
                    "[1] https://example.com/source-a"
                )
            )
        ]
    )
    monkeypatch.setattr(report, "get_llm", lambda role: model)

    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": ["Finding [1] https://example.com/source-a"],
                "raw_notes": ["Raw [1] https://example.com/source-a"],
                "final_report": "",
            }
        )
    )

    assert "Sources:" in result["final_report"]
    assert "[1]" in result["final_report"]
    assert result["messages"][-1].content == result["final_report"]
    assert len(model.calls) == 1
    assert model.calls[0][0].type == "system"
    assert model.calls[0][1].type == "human"
    assert "Research brief:\nBrief" in model.calls[0][1].content


def test_final_report_generation_token_limit_retry_path_is_deterministic(monkeypatch):
    model = _FakeReportModel(
        [
            RuntimeError("maximum context length exceeded"),
            AIMessage(content="Recovered analysis [1]"),
        ]
    )
    monkeypatch.setattr(report, "get_llm", lambda role: model)

    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": ["Finding A [1]", "Finding B [2]", "Finding C [3]"],
                "raw_notes": [
                    "https://example.com/source-a",
                    "https://example.com/source-b",
                    "https://example.com/source-c",
                ],
                "final_report": "",
            }
        )
    )

    assert len(model.calls) == 2
    first_prompt = model.calls[0][1].content
    second_prompt = model.calls[1][1].content
    assert model.calls[0][0].type == "system"
    assert model.calls[1][0].type == "system"
    assert "Finding C [3]" in first_prompt
    assert "Finding C [3]" not in second_prompt
    assert "Recovered analysis [1]" in result["final_report"]
    assert "Sources:" in result["final_report"]
    assert "https://example.com/source-a" in result["final_report"]


def test_final_report_generation_no_notes_falls_back_with_source_transparency(monkeypatch):
    model = _FakeReportModel([AIMessage(content=""), AIMessage(content=""), AIMessage(content="")])
    monkeypatch.setattr(report, "get_llm", lambda role: model)

    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": [],
                "raw_notes": [],
                "final_report": "",
            }
        )
    )

    assert len(model.calls) == 3
    assert FALLBACK_FINAL_REPORT in result["final_report"]
    assert "Sources:" in result["final_report"]
    assert "No source URLs were available in collected notes." in result["final_report"]


def test_final_report_generation_filters_malformed_urls_from_sources(monkeypatch):
    model = _FakeReportModel([AIMessage(content="Summary with evidence [1].")])
    monkeypatch.setattr(report, "get_llm", lambda role: model)

    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": [
                    "Valid: https://example.com/source-a",
                    "Truncated: https://example.com/source-b-",
                ],
                "raw_notes": ["Missing host: https:///missing-netloc"],
                "evidence_ledger": [
                    {"source_urls": ["https://example.org/source-c", "https://example.net/source-d-"]},
                ],
                "final_report": "",
            }
        )
    )

    assert "Sources:" in result["final_report"]
    assert "https://example.com/source-a" in result["final_report"]
    assert "https://example.org/source-c" in result["final_report"]
    assert "https://example.com/source-b-" not in result["final_report"]
    assert "https://example.net/source-d-" not in result["final_report"]
    assert "https:///missing-netloc" not in result["final_report"]


def test_final_report_generation_does_not_duplicate_existing_markdown_sources_heading(monkeypatch):
    model = _FakeReportModel(
        [
            AIMessage(
                content=(
                    "## Summary\n"
                    "Findings with citation [1].\n\n"
                    "## Sources\n"
                    "[1] https://example.com/source-a"
                )
            )
        ]
    )
    monkeypatch.setattr(report, "get_llm", lambda role: model)

    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": ["Finding [1] https://example.com/source-a"],
                "raw_notes": [],
                "final_report": "",
            }
        )
    )

    assert result["final_report"].count("Sources") == 1
    assert "No source URLs were available in collected notes." not in result["final_report"]


def test_final_report_generation_removes_contradictory_no_sources_line_when_urls_exist(monkeypatch):
    model = _FakeReportModel(
        [
            AIMessage(
                content=(
                    "Summary [1].\n\n"
                    "Sources:\n"
                    "[1] https://example.com/source-a\n"
                    "- No source URLs were available in collected notes."
                )
            )
        ]
    )
    monkeypatch.setattr(report, "get_llm", lambda role: model)

    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": ["Finding [1] https://example.com/source-a"],
                "raw_notes": [],
                "final_report": "",
            }
        )
    )

    assert "https://example.com/source-a" in result["final_report"]
    assert "No source URLs were available in collected notes." not in result["final_report"]


def test_final_report_generation_strips_internal_tool_meta_lines(monkeypatch):
    model = _FakeReportModel(
        [
            AIMessage(
                content=(
                    "## Findings\n"
                    "Market growth accelerated in 2025 [1].\n"
                    "I used ConductResearch twice and then ran search_web.\n"
                    "[ConductResearch skipped: reached runtime cap]\n"
                    "Reflection recorded: need another pass.\n"
                    "Sources:\n"
                    "[1] https://example.com/source-a"
                )
            )
        ]
    )
    monkeypatch.setattr(report, "get_llm", lambda role: model)

    result = asyncio.run(
        report.final_report_generation(
            {
                "research_brief": "Brief",
                "notes": ["Finding [1] https://example.com/source-a"],
                "raw_notes": [],
                "final_report": "",
            }
        )
    )

    assert "Market growth accelerated in 2025 [1]." in result["final_report"]
    assert "ConductResearch" not in result["final_report"]
    assert "search_web" not in result["final_report"]
    assert "Reflection recorded" not in result["final_report"]
