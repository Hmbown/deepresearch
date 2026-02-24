from deepresearch import prompts
from deepresearch.state import today_utc_date

import pytest

TEST_DATE = today_utc_date()


def test_clarify_prompt_includes_multi_turn_instruction():
    required_tokens = [
        "<Messages>",
        "{messages}",
        "{date}",
        "clarif",
        "scope",
    ]
    for token in required_tokens:
        assert token in prompts.CLARIFY_PROMPT


def test_research_brief_prompt_has_history_and_specificity_contract():
    required_tokens = [
        "<Messages>",
        "{messages}",
        "{date}",
        "research brief",
    ]
    for token in required_tokens:
        assert token in prompts.RESEARCH_BRIEF_PROMPT


def test_supervisor_prompt_is_native_multi_agent_contract():
    required_tokens = [
        "ConductResearch",
        "ResearchComplete",
        "think_tool",
        "max_concurrent_research_units",
        "max_researcher_iterations",
        "same language",
        "citation",
    ]
    for token in required_tokens:
        assert token in prompts.SUPERVISOR_PROMPT

    forbidden_tokens = ["write_todos", "task(", "subagent_type=\"research-agent\""]
    for token in forbidden_tokens:
        assert token not in prompts.SUPERVISOR_PROMPT


def test_researcher_prompt_preserves_research_quality_contract():
    required_tokens = [
        "search_web",
        "fetch_url",
        "think_tool",
        "Contradictions/Uncertainties",
        "citation",
        "max_react_tool_calls",
        "same language",
    ]
    for token in required_tokens:
        assert token in prompts.RESEARCHER_PROMPT


def test_final_report_prompt_has_required_placeholders():
    assert "{current_date}" in prompts.FINAL_REPORT_PROMPT


def test_final_report_prompt_enforces_citation_and_url_quality_rules():
    required_tokens = [
        "Cite only factual claims about the world",
        "Prefer SEC-hosted filing URLs",
        "generic SEC EDGAR search pages",
        "USCourts pages",
        "complete and not truncated",
        'include an "as of" date',
    ]
    for token in required_tokens:
        assert token in prompts.FINAL_REPORT_PROMPT


def test_research_plan_prompt_has_required_placeholders():
    assert "{research_brief}" in prompts.RESEARCH_PLAN_PROMPT
    assert "{date}" in prompts.RESEARCH_PLAN_PROMPT
    assert "{max_research_tracks}" in prompts.RESEARCH_PLAN_PROMPT
    assert prompts.RESEARCH_PLAN_PROMPT.format(research_brief="brief", date=TEST_DATE, max_research_tracks=4)


def test_prompt_templates_render_with_expected_keys_and_fail_closed_on_missing_fields():
    assert prompts.CLARIFY_PROMPT.format(messages="m", date=TEST_DATE)
    assert prompts.RESEARCH_BRIEF_PROMPT.format(messages="m", date=TEST_DATE)
    assert prompts.SUPERVISOR_PROMPT.format(
        current_date=TEST_DATE,
        max_concurrent_research_units=4,
        max_researcher_iterations=6,
    )
    assert prompts.RESEARCHER_PROMPT.format(
        max_react_tool_calls=10,
    )
    assert prompts.FINAL_REPORT_PROMPT.format(current_date=TEST_DATE)
    assert prompts.RESEARCH_PLAN_PROMPT.format(research_brief="brief", date=TEST_DATE, max_research_tracks=4)

    with pytest.raises(KeyError):
        prompts.CLARIFY_PROMPT.format(messages="m")
    with pytest.raises(KeyError):
        prompts.RESEARCH_BRIEF_PROMPT.format(messages="m")
    with pytest.raises(KeyError):
        prompts.SUPERVISOR_PROMPT.format(current_date=TEST_DATE)
    with pytest.raises(KeyError):
        prompts.RESEARCHER_PROMPT.format()


def test_researcher_prompt_sections_are_explicit_and_ordered():
    section_markers = [
        "Executive Summary",
        "Key Findings",
        "Evidence Log",
        "Contradictions/Uncertainties",
        "Gaps/Next Questions",
    ]
    positions = [prompts.RESEARCHER_PROMPT.find(marker) for marker in section_markers]
    assert all(pos >= 0 for pos in positions)
    assert positions == sorted(positions)
