import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

from deepresearch import cli, config


def test_resolve_model_for_role_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_MODEL", raising=False)
    monkeypatch.delenv("SUBAGENT_MODEL", raising=False)

    assert config._resolve_model_for_role("orchestrator") == config.DEFAULT_ORCHESTRATOR_MODEL
    assert config._resolve_model_for_role("subagent") == config.DEFAULT_SUBAGENT_MODEL
    assert config._resolve_model_for_role("unknown_role") == config.DEFAULT_SUBAGENT_MODEL

    monkeypatch.setenv("ORCHESTRATOR_MODEL", "openai:o-test")
    monkeypatch.setenv("SUBAGENT_MODEL", "openai:s-test")
    assert config._resolve_model_for_role("orchestrator") == "openai:o-test"
    assert config._resolve_model_for_role("subagent") == "openai:s-test"
    assert config._resolve_model_for_role("unknown_role") == "openai:s-test"


def test_get_llm_passes_resolved_model(monkeypatch):
    captured = {}

    def fake_init_chat_model(*, model):
        captured["model"] = model
        return {"model": model}

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "openai:gpt-test-52")
    monkeypatch.setenv("SUBAGENT_MODEL", "openai:gpt-test-mini")

    llm = config.get_llm("subagent")
    assert llm == {"model": "openai:gpt-test-mini"}
    assert captured["model"] == "openai:gpt-test-mini"


def test_int_env_config_keys_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("MAX_STRUCTURED_OUTPUT_RETRIES", "0")
    monkeypatch.setenv("MAX_REACT_TOOL_CALLS", "0")
    monkeypatch.setenv("MAX_CONCURRENT_RESEARCH_UNITS", "-3")
    monkeypatch.setenv("MAX_RESEARCHER_ITERATIONS", "-2")
    monkeypatch.setenv("RESEARCHER_SIMPLE_SEARCH_BUDGET", "abc")
    monkeypatch.setenv("RESEARCHER_COMPLEX_SEARCH_BUDGET", "-1")
    monkeypatch.setenv("SUPERVISOR_NOTES_MAX_BULLETS", "0")
    monkeypatch.setenv("SUPERVISOR_NOTES_WORD_BUDGET", "10")
    monkeypatch.setenv("SUPERVISOR_FINAL_REPORT_MAX_SECTIONS", "-10")

    assert config.get_max_structured_output_retries() == config.DEFAULT_MAX_STRUCTURED_OUTPUT_RETRIES
    assert config.get_max_react_tool_calls() == config.DEFAULT_MAX_REACT_TOOL_CALLS
    assert config.get_max_concurrent_research_units() == config.DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS
    assert config.get_max_researcher_iterations() == config.DEFAULT_MAX_RESEARCHER_ITERATIONS
    assert config.get_researcher_simple_search_budget() == config.DEFAULT_RESEARCHER_SIMPLE_SEARCH_BUDGET
    assert config.get_researcher_complex_search_budget() == config.DEFAULT_RESEARCHER_COMPLEX_SEARCH_BUDGET
    assert config.get_supervisor_notes_max_bullets() == config.DEFAULT_SUPERVISOR_NOTES_MAX_BULLETS
    assert config.get_supervisor_notes_word_budget() == config.DEFAULT_SUPERVISOR_NOTES_WORD_BUDGET
    assert (
        config.get_supervisor_final_report_max_sections()
        == config.DEFAULT_SUPERVISOR_FINAL_REPORT_MAX_SECTIONS
    )


def test_runtime_env_overrides_are_respected(monkeypatch):
    monkeypatch.setenv("MAX_STRUCTURED_OUTPUT_RETRIES", "5")
    monkeypatch.setenv("MAX_REACT_TOOL_CALLS", "8")
    monkeypatch.setenv("MAX_CONCURRENT_RESEARCH_UNITS", "2")
    monkeypatch.setenv("MAX_RESEARCHER_ITERATIONS", "7")
    monkeypatch.setenv("SUPERVISOR_NOTES_MAX_BULLETS", "12")
    monkeypatch.setenv("SUPERVISOR_NOTES_WORD_BUDGET", "350")
    monkeypatch.setenv("SUPERVISOR_FINAL_REPORT_MAX_SECTIONS", "5")

    assert config.get_max_structured_output_retries() == 5
    assert config.get_max_react_tool_calls() == 8
    assert config.get_max_concurrent_research_units() == 2
    assert config.get_max_researcher_iterations() == 7
    assert config.get_supervisor_notes_max_bullets() == 12
    assert config.get_supervisor_notes_word_budget() == 350
    assert config.get_supervisor_final_report_max_sections() == 5


def test_get_search_tool_prefers_exa_over_tavily(monkeypatch):
    class FakeTavilySearch:
        def __init__(self, **kwargs):
            self.provider = "tavily"
            self.kwargs = kwargs

    class FakeExaSearchResults:
        def __init__(self, **kwargs):
            self.provider = "exa"
            self.kwargs = kwargs

    fake_langchain_tavily = types.SimpleNamespace(TavilySearch=FakeTavilySearch)
    fake_langchain_exa = types.SimpleNamespace(ExaSearchResults=FakeExaSearchResults)

    monkeypatch.setenv("TAVILY_API_KEY", "tav-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setitem(sys.modules, "langchain_tavily", fake_langchain_tavily)
    monkeypatch.setitem(sys.modules, "langchain_exa", fake_langchain_exa)

    tool = config.get_search_tool()
    assert tool.provider == "exa"
    assert tool.kwargs["exa_api_key"] == "exa-key"
    assert "max_results" not in tool.kwargs


def test_get_search_tool_returns_none_when_exa_unavailable(monkeypatch):
    class FakeTavilySearch:
        def __init__(self, **kwargs):
            self.provider = "tavily"
            self.kwargs = kwargs

    fake_langchain_exa = None
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tav-key")
    monkeypatch.setitem(sys.modules, "langchain_exa", fake_langchain_exa)

    tool = config.get_search_tool()
    assert tool is None


def test_get_search_tool_returns_none_with_no_api_keys_or_tools(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert config.get_search_tool() is None


def test_get_search_tool_uses_none_when_exa_unset(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tav-key")

    tool = config.get_search_tool()
    assert tool is None


def test_get_search_tool_uses_exa_when_tavily_unset(monkeypatch):
    class FakeExaSearchResults:
        def __init__(self, **kwargs):
            self.provider = "exa"
            self.kwargs = kwargs

    fake_langchain_exa = types.SimpleNamespace(ExaSearchResults=FakeExaSearchResults)

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setitem(sys.modules, "langchain_exa", fake_langchain_exa)

    tool = config.get_search_tool()
    assert tool.provider == "exa"
    assert tool.kwargs["exa_api_key"] == "exa-key"
    assert "max_results" not in tool.kwargs


def test_search_budget_knobs_parse_and_default(monkeypatch):
    monkeypatch.delenv("RESEARCHER_SIMPLE_SEARCH_BUDGET", raising=False)
    monkeypatch.delenv("RESEARCHER_COMPLEX_SEARCH_BUDGET", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_RESEARCH_UNITS", raising=False)
    monkeypatch.delenv("MAX_RESEARCHER_ITERATIONS", raising=False)

    assert config.get_researcher_simple_search_budget() == config.DEFAULT_RESEARCHER_SIMPLE_SEARCH_BUDGET
    assert config.get_researcher_complex_search_budget() == config.DEFAULT_RESEARCHER_COMPLEX_SEARCH_BUDGET
    assert config.get_max_concurrent_research_units() == config.DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS
    assert config.get_max_researcher_iterations() == config.DEFAULT_MAX_RESEARCHER_ITERATIONS

    monkeypatch.setenv("RESEARCHER_SIMPLE_SEARCH_BUDGET", "7")
    monkeypatch.setenv("RESEARCHER_COMPLEX_SEARCH_BUDGET", "11")
    monkeypatch.setenv("MAX_CONCURRENT_RESEARCH_UNITS", "3")
    monkeypatch.setenv("MAX_RESEARCHER_ITERATIONS", "9")
    assert config.get_researcher_simple_search_budget() == 7
    assert config.get_researcher_complex_search_budget() == 11
    assert config.get_max_concurrent_research_units() == 3
    assert config.get_max_researcher_iterations() == 9


def test_cli_extract_text_content_from_mixed_blocks():
    text = cli._extract_text_content(
        [
            {"type": "text", "text": "line 1"},
            {"type": "input_text", "text": "ignored"},
            "line 2",
        ]
    )
    assert text == "line 1\nline 2"


def test_cli_final_assistant_text_finds_last_ai_message():
    result = {
        "messages": [
            SimpleNamespace(type="human", content="question"),
            SimpleNamespace(type="ai", content="first"),
            SimpleNamespace(type="ai", content=[{"type": "text", "text": "final"}]),
        ]
    }
    assert cli._final_assistant_text(result) == "final"


def test_cli_final_assistant_text_prefers_final_report_field():
    result = {
        "final_report": "structured final report",
        "messages": [SimpleNamespace(type="ai", content="fallback message")],
    }
    assert cli._final_assistant_text(result) == "structured final report"


def test_cli_run_invokes_app_with_human_message(monkeypatch):
    fake_app = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": []}))
    monkeypatch.setattr(cli, "_get_app", lambda: fake_app)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    asyncio.run(cli.run("test query", thread_id="thread-test"))
    assert fake_app.ainvoke.await_count == 1
    payload = fake_app.ainvoke.await_args.args[0]
    config_payload = fake_app.ainvoke.await_args.kwargs["config"]
    assert "messages" in payload
    assert payload["messages"][0].type == "human"
    assert payload["messages"][0].content == "test query"
    assert config_payload == {"configurable": {"thread_id": "thread-test"}}


def test_cli_run_appends_prior_messages_when_provided(monkeypatch):
    fake_app = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": []}))
    monkeypatch.setattr(cli, "_get_app", lambda: fake_app)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    prior = [SimpleNamespace(type="human", content="existing context")]
    asyncio.run(cli.run("next query", thread_id="thread-test", prior_messages=prior))

    payload = fake_app.ainvoke.await_args.args[0]
    assert len(payload["messages"]) == 2
    assert payload["messages"][0].content == "existing context"
    assert payload["messages"][1].content == "next query"


def test_cli_run_generates_thread_id_when_missing(monkeypatch):
    fake_app = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": []}))
    monkeypatch.setattr(cli, "_get_app", lambda: fake_app)
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: SimpleNamespace(hex="generated-thread"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    asyncio.run(cli.run("query without thread"))
    assert fake_app.ainvoke.await_count == 1
    config_payload = fake_app.ainvoke.await_args.kwargs["config"]
    assert config_payload == {"configurable": {"thread_id": "generated-thread"}}


def test_cli_result_section_title_uses_clarification_label():
    assert cli._result_section_title({"intake_decision": "clarify"}) == "CLARIFICATION"
    assert cli._result_section_title({"intake_decision": "proceed"}) == "RESPONSE"


def test_cli_result_section_title_defaults_to_response_without_decision():
    assert cli._result_section_title({}) == "RESPONSE"


def test_cli_print_results_prints_no_response_message_when_ai_missing(capsys):
    cli.print_results({"messages": [SimpleNamespace(type="human", content="query")]})
    output = capsys.readouterr().out
    assert output.strip() == "No assistant response found."
