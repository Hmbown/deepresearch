from deepresearch import prompts

import pytest


def test_clarify_prompt_includes_multi_turn_instruction():
    required_tokens = [
        "<Messages>",
        "{messages}",
        "{date}",
        "ask ONE clarifying question",
        "already asked a clarifying question",
        "almost always proceed",
        "Verification message rules",
    ]
    for token in required_tokens:
        assert token in prompts.CLARIFY_PROMPT


def test_research_brief_prompt_has_history_and_specificity_contract():
    required_tokens = [
        "<Messages>",
        "{messages}",
        "{date}",
        "focused research brief",
        "Maximize specificity and detail",
        "Avoid unwarranted assumptions",
        "primary and official sources",
    ]
    for token in required_tokens:
        assert token in prompts.RESEARCH_BRIEF_PROMPT


def test_supervisor_prompt_is_native_multi_agent_contract():
    required_tokens = [
        "ConductResearch(research_topic)",
        "ResearchComplete()",
        "think_tool",
        "MAX_CONCURRENT_RESEARCH_UNITS",
        "MAX_RESEARCHER_ITERATIONS",
        "same language as the user request",
        "inline citations [1], [2]",
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
        "Evidence targets",
        "Search budget guidance",
        "Contradictions/Uncertainties",
        "inline citation numbers [1], [2]",
        "researcher_simple_search_budget",
        "researcher_complex_search_budget",
        "max_react_tool_calls",
        "same language as the user request",
    ]
    for token in required_tokens:
        assert token in prompts.RESEARCHER_PROMPT


def test_compression_and_final_report_prompts_have_required_placeholders():
    assert "{notes_max_bullets}" in prompts.COMPRESSION_PROMPT
    assert "{notes_word_budget}" in prompts.COMPRESSION_PROMPT
    assert "{current_date}" in prompts.FINAL_REPORT_PROMPT
    assert "{final_report_max_sections}" in prompts.FINAL_REPORT_PROMPT


def test_prompt_templates_render_with_expected_keys_and_fail_closed_on_missing_fields():
    assert prompts.CLARIFY_PROMPT.format(messages="m", date="2026-02-23")
    assert prompts.RESEARCH_BRIEF_PROMPT.format(messages="m", date="2026-02-23")
    assert prompts.SUPERVISOR_PROMPT.format(
        current_date="2026-02-23",
        max_concurrent_research_units=4,
        max_researcher_iterations=6,
    )
    assert prompts.RESEARCHER_PROMPT.format(
        researcher_simple_search_budget=3,
        researcher_complex_search_budget=5,
        max_react_tool_calls=10,
    )
    assert prompts.COMPRESSION_PROMPT.format(notes_max_bullets=8, notes_word_budget=250)
    assert prompts.FINAL_REPORT_PROMPT.format(current_date="2026-02-23", final_report_max_sections=8)

    with pytest.raises(KeyError):
        prompts.CLARIFY_PROMPT.format(messages="m")
    with pytest.raises(KeyError):
        prompts.RESEARCH_BRIEF_PROMPT.format(messages="m")
    with pytest.raises(KeyError):
        prompts.SUPERVISOR_PROMPT.format(current_date="2026-02-23")
    with pytest.raises(KeyError):
        prompts.RESEARCHER_PROMPT.format(researcher_simple_search_budget=3)


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
