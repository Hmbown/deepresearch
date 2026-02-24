import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deepresearch import cli, config, env


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

    def fake_init_chat_model(**kwargs):
        captured["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "openai:gpt-test-52")
    monkeypatch.setenv("SUBAGENT_MODEL", "openai:gpt-test-mini")
    monkeypatch.setenv("OPENAI_USE_RESPONSES_API", "false")

    llm = config.get_llm("subagent")
    assert llm == {"model": "openai:gpt-test-mini"}
    assert captured["kwargs"] == {"model": "openai:gpt-test-mini"}


def test_openai_responses_api_defaults_enabled(monkeypatch):
    monkeypatch.delenv("OPENAI_USE_RESPONSES_API", raising=False)
    assert config.openai_responses_api_enabled() is True


def test_openai_responses_api_explicit_opt_out(monkeypatch):
    """Verify explicit opt-out via OPENAI_USE_RESPONSES_API=false disables Responses API."""
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "openai:gpt-test")
    monkeypatch.setenv("OPENAI_USE_RESPONSES_API", "false")

    config.get_llm("orchestrator")
    assert "use_responses_api" not in captured["kwargs"]
    assert captured["kwargs"] == {"model": "openai:gpt-test"}


def test_openai_responses_api_non_openai_provider_unaffected(monkeypatch):
    """Responses API flags should not be passed for non-OpenAI providers."""
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "anthropic:claude-opus-4-6")
    monkeypatch.delenv("OPENAI_USE_RESPONSES_API", raising=False)

    config.get_llm("orchestrator")
    assert "use_responses_api" not in captured["kwargs"]
    assert captured["kwargs"] == {"model": "anthropic:claude-opus-4-6"}


def test_get_llm_uses_openai_responses_api_by_default(monkeypatch):
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("SUBAGENT_MODEL", "openai:gpt-test-mini")
    monkeypatch.delenv("OPENAI_USE_RESPONSES_API", raising=False)
    monkeypatch.delenv("OPENAI_USE_PREVIOUS_RESPONSE_ID", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_VERSION", raising=False)

    llm = config.get_llm("subagent")
    assert llm["model"] == "openai:gpt-test-mini"
    assert llm["use_responses_api"] is True
    assert llm["output_version"] == "responses/v1"
    assert "use_previous_response_id" not in llm
    assert captured["kwargs"] == llm


def test_get_llm_can_enable_openai_responses_api(monkeypatch):
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured["kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("SUBAGENT_MODEL", "openai:gpt-test-mini")
    monkeypatch.setenv("OPENAI_USE_RESPONSES_API", "true")
    monkeypatch.setenv("OPENAI_OUTPUT_VERSION", "responses/v1")
    monkeypatch.setenv("OPENAI_USE_PREVIOUS_RESPONSE_ID", "true")

    llm = config.get_llm("subagent")
    assert llm == {
        "model": "openai:gpt-test-mini",
        "use_responses_api": True,
        "output_version": "responses/v1",
        "use_previous_response_id": True,
    }
    assert captured["kwargs"] == llm


def test_get_llm_openai_responses_flag_falls_back_when_kwargs_unsupported(monkeypatch):
    calls: list[dict] = []

    def fake_init_chat_model(**kwargs):
        calls.append(kwargs)
        if "use_responses_api" in kwargs:
            raise TypeError("unexpected keyword argument 'use_responses_api'")
        return kwargs

    monkeypatch.setattr(config, "init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "openai:gpt-test-52")
    monkeypatch.setenv("OPENAI_USE_RESPONSES_API", "true")

    llm = config.get_llm("orchestrator")
    assert llm == {"model": "openai:gpt-test-52"}
    assert calls[0]["use_responses_api"] is True
    assert calls[1] == {"model": "openai:gpt-test-52"}


def test_int_env_config_keys_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("MAX_STRUCTURED_OUTPUT_RETRIES", "0")
    monkeypatch.setenv("MAX_REACT_TOOL_CALLS", "0")
    monkeypatch.setenv("MAX_CONCURRENT_RESEARCH_UNITS", "-3")
    monkeypatch.setenv("MAX_RESEARCHER_ITERATIONS", "-2")
    monkeypatch.setenv("MAX_EVIDENCE_CLAIMS_PER_RESEARCH_UNIT", "0")
    monkeypatch.setenv("RESEARCHER_SEARCH_BUDGET", "abc")
    monkeypatch.setenv("MAX_SOURCE_URLS_PER_CLAIM", "0")
    monkeypatch.setenv("SUPERVISOR_NOTES_MAX_BULLETS", "0")
    monkeypatch.setenv("SUPERVISOR_NOTES_WORD_BUDGET", "10")
    monkeypatch.setenv("SUPERVISOR_FINAL_REPORT_MAX_SECTIONS", "-10")

    assert config.get_max_structured_output_retries() == config.DEFAULT_MAX_STRUCTURED_OUTPUT_RETRIES
    assert config.get_max_react_tool_calls() == config.DEFAULT_MAX_REACT_TOOL_CALLS
    assert config.get_max_concurrent_research_units() == config.DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS
    assert config.get_max_researcher_iterations() == config.DEFAULT_MAX_RESEARCHER_ITERATIONS
    assert (
        config.get_max_evidence_claims_per_research_unit()
        == config.DEFAULT_MAX_EVIDENCE_CLAIMS_PER_RESEARCH_UNIT
    )
    assert config.get_max_source_urls_per_claim() == config.DEFAULT_MAX_SOURCE_URLS_PER_CLAIM
    assert config.get_researcher_search_budget() == config.DEFAULT_RESEARCHER_SEARCH_BUDGET
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
    monkeypatch.setenv("MAX_EVIDENCE_CLAIMS_PER_RESEARCH_UNIT", "6")
    monkeypatch.setenv("MAX_SOURCE_URLS_PER_CLAIM", "12")
    monkeypatch.setenv("SUPERVISOR_NOTES_MAX_BULLETS", "12")
    monkeypatch.setenv("SUPERVISOR_NOTES_WORD_BUDGET", "350")
    monkeypatch.setenv("SUPERVISOR_FINAL_REPORT_MAX_SECTIONS", "5")

    assert config.get_max_structured_output_retries() == 5
    assert config.get_max_react_tool_calls() == 8
    assert config.get_max_concurrent_research_units() == 2
    assert config.get_max_researcher_iterations() == 7
    assert config.get_max_evidence_claims_per_research_unit() == 6
    assert config.get_max_source_urls_per_claim() == 12
    assert config.get_supervisor_notes_max_bullets() == 12
    assert config.get_supervisor_notes_word_budget() == 350
    assert config.get_supervisor_final_report_max_sections() == 5


def test_get_search_provider_defaults_to_exa(monkeypatch):
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    assert config.get_search_provider() == "exa"


def test_get_search_provider_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "duckduckgo")
    with pytest.raises(config.SearchProviderConfigError, match="Invalid SEARCH_PROVIDER"):
        config.get_search_provider()


def test_validate_search_provider_configuration_fails_for_missing_exa_key(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "exa")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(config.SearchProviderConfigError, match="EXA_API_KEY"):
        config.validate_search_provider_configuration()


def test_validate_search_provider_configuration_fails_for_missing_tavily_key(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(config.SearchProviderConfigError, match="TAVILY_API_KEY"):
        config.validate_search_provider_configuration()


def test_validate_search_provider_configuration_fails_for_missing_exa_dependency(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "exa")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setitem(sys.modules, "langchain_exa", None)
    with pytest.raises(config.SearchProviderConfigError, match="langchain-exa"):
        config.validate_search_provider_configuration()


def test_validate_search_provider_configuration_fails_for_missing_tavily_dependency(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setitem(sys.modules, "langchain_tavily", None)
    with pytest.raises(config.SearchProviderConfigError, match="langchain-tavily"):
        config.validate_search_provider_configuration()


def test_get_search_tool_uses_exa_when_provider_is_configured(monkeypatch):
    class FakeExaSearchResults:
        def __init__(self, **kwargs):
            self.provider = "exa"
            self.kwargs = kwargs

    fake_langchain_exa = types.SimpleNamespace(ExaSearchResults=FakeExaSearchResults)

    monkeypatch.setenv("SEARCH_PROVIDER", "exa")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    monkeypatch.setitem(sys.modules, "langchain_exa", fake_langchain_exa)

    tool = config.get_search_tool()
    assert tool.provider == "exa"
    assert tool.kwargs["exa_api_key"] == "exa-key"


def test_get_search_tool_uses_tavily_when_provider_is_configured(monkeypatch):
    class FakeTavilySearchResults:
        def __init__(self, **kwargs):
            self.provider = "tavily"
            self.kwargs = kwargs

    fake_langchain_tavily = types.SimpleNamespace(TavilySearchResults=FakeTavilySearchResults)

    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setitem(sys.modules, "langchain_tavily", fake_langchain_tavily)

    tool = config.get_search_tool()
    assert tool.provider == "tavily"
    assert tool.kwargs["api_key"] == "tavily-key"


def test_get_search_tool_returns_none_when_provider_is_none(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    tool = config.get_search_tool()
    assert tool is None


def test_runtime_event_logs_enabled_reads_env_truthiness(monkeypatch):
    monkeypatch.delenv("ENABLE_RUNTIME_EVENT_LOGS", raising=False)
    assert config.runtime_event_logs_enabled() is False
    monkeypatch.setenv("ENABLE_RUNTIME_EVENT_LOGS", "true")
    assert config.runtime_event_logs_enabled() is True


def test_search_budget_knobs_parse_and_default(monkeypatch):
    monkeypatch.delenv("RESEARCHER_SEARCH_BUDGET", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_RESEARCH_UNITS", raising=False)
    monkeypatch.delenv("MAX_RESEARCHER_ITERATIONS", raising=False)

    assert config.get_researcher_search_budget() == config.DEFAULT_RESEARCHER_SEARCH_BUDGET
    assert config.get_max_concurrent_research_units() == config.DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS
    assert config.get_max_researcher_iterations() == config.DEFAULT_MAX_RESEARCHER_ITERATIONS

    monkeypatch.setenv("RESEARCHER_SEARCH_BUDGET", "9")
    monkeypatch.setenv("MAX_CONCURRENT_RESEARCH_UNITS", "3")
    monkeypatch.setenv("MAX_RESEARCHER_ITERATIONS", "9")
    assert config.get_researcher_search_budget() == 9
    assert config.get_max_concurrent_research_units() == 3
    assert config.get_max_researcher_iterations() == 9


def test_runtime_defaults_use_extended_profile_values():
    assert config.DEFAULT_RESEARCHER_SEARCH_BUDGET == 15
    assert config.DEFAULT_MAX_REACT_TOOL_CALLS == 40
    assert config.DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS == 4
    assert config.DEFAULT_MAX_RESEARCHER_ITERATIONS == 60
    assert config.DEFAULT_MAX_EVIDENCE_CLAIMS_PER_RESEARCH_UNIT == 5
    assert config.DEFAULT_MAX_SOURCE_URLS_PER_CLAIM == 5
    assert config.DEFAULT_SUPERVISOR_NOTES_MAX_BULLETS == 40
    assert config.DEFAULT_SUPERVISOR_NOTES_WORD_BUDGET == 1200
    assert config.DEFAULT_SUPERVISOR_FINAL_REPORT_MAX_SECTIONS == 12


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
    monkeypatch.setenv("SEARCH_PROVIDER", "none")

    asyncio.run(cli.run("test query", thread_id="thread-test"))
    assert fake_app.ainvoke.await_count == 1
    payload = fake_app.ainvoke.await_args.args[0]
    config_payload = fake_app.ainvoke.await_args.kwargs["config"]
    assert "messages" in payload
    assert payload["messages"][0].type == "human"
    assert payload["messages"][0].content == "test query"
    assert config_payload == {"configurable": {"thread_id": "thread-test"}, "recursion_limit": 1000}


def test_cli_run_appends_prior_messages_when_provided(monkeypatch):
    fake_app = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": []}))
    monkeypatch.setattr(cli, "_get_app", lambda: fake_app)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")

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
    monkeypatch.setenv("SEARCH_PROVIDER", "none")

    asyncio.run(cli.run("query without thread"))
    assert fake_app.ainvoke.await_count == 1
    config_payload = fake_app.ainvoke.await_args.kwargs["config"]
    assert config_payload == {"configurable": {"thread_id": "generated-thread"}, "recursion_limit": 1000}


def test_cli_result_section_title_uses_clarification_label():
    assert cli._result_section_title({"intake_decision": "clarify"}) == "CLARIFICATION"
    assert cli._result_section_title(
        {"intake_decision": "clarify"},
        elapsed_seconds=12.0,
    ) == "CLARIFICATION (12s)"


def test_cli_result_section_title_uses_research_plan_label_for_plan_checkpoint():
    result = {
        "intake_decision": "clarify",
        "messages": [
            SimpleNamespace(
                type="ai",
                content=(
                    'Before I start research, here is the plan.\n'
                    'If this plan looks right, reply "start" to launch research.'
                ),
            )
        ],
    }
    assert cli._result_section_title(result) == "RESEARCH PLAN"
    assert cli._result_section_title(result, elapsed_seconds=12.0) == "RESEARCH PLAN (12s)"


def test_cli_result_section_title_defaults_to_research_report_with_stats():
    result = {
        "evidence_ledger": [
            {"claim": "A", "source_urls": ["https://example.com/a"], "confidence": 0.5},
            {"claim": "B", "source_urls": ["https://example.org/b"], "confidence": 0.5},
        ]
    }
    assert cli._result_section_title(result) == "RESEARCH REPORT (2 evidence records | 2 sources)"
    assert cli._result_section_title(result, elapsed_seconds=61.0) == (
        "RESEARCH REPORT (2 evidence records | 2 sources | 1m 01s)"
    )


def test_cli_print_results_prints_no_response_message_when_ai_missing(capsys):
    cli.print_results({"messages": [SimpleNamespace(type="human", content="query")]})
    output = capsys.readouterr().out
    assert output.strip() == "No assistant response found."


def test_cli_main_help_uses_argparse(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["deepresearch", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0


def test_cli_main_preflight_flag(monkeypatch):
    called = {}

    def fake_print_preflight(project_name=None):
        called["project_name"] = project_name
        return 0

    monkeypatch.setattr(cli, "print_preflight", fake_print_preflight)
    monkeypatch.setattr(sys, "argv", ["deepresearch", "--preflight", "demo-project"])
    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert called["project_name"] == "demo-project"


def test_cli_main_setup_flag(monkeypatch):
    called = {"count": 0}

    def fake_run_setup_wizard():
        called["count"] += 1
        return 0

    monkeypatch.setattr(cli, "run_setup_wizard", fake_run_setup_wizard)
    monkeypatch.setattr(sys, "argv", ["deepresearch", "--setup"])
    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert called["count"] == 1


def test_cli_main_setup_rejects_query(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["deepresearch", "--setup", "why now"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_cli_main_setup_rejects_combination_with_preflight(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["deepresearch", "--setup", "--preflight"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_setup_wizard_writes_exa_path_and_runs_preflight(monkeypatch, tmp_path, capsys):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("UNRELATED_KEY=keep\nTAVILY_API_KEY=keep-tavily\n", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setattr(env, "_BOOTSTRAPPED_DOTENV", None)
    prompts = iter(["exa", "n"])
    secrets = iter(["openai-key-123", "exa-key-123"])
    preflight_calls: list[str | None] = []

    monkeypatch.setattr("builtins.input", lambda _="": next(prompts))
    monkeypatch.setattr(cli, "getpass", lambda _="": next(secrets))
    monkeypatch.setattr(cli, "print_preflight", lambda project_name=None: preflight_calls.append(project_name) or 0)

    result = cli.run_setup_wizard()

    output = capsys.readouterr().out
    contents = dotenv_path.read_text(encoding="utf-8")
    assert result == 0
    assert preflight_calls == [None]
    assert "OPENAI_API_KEY=openai-key-123" in contents
    assert "SEARCH_PROVIDER=exa" in contents
    assert "EXA_API_KEY=exa-key-123" in contents
    assert "LANGCHAIN_TRACING_V2=false" in contents
    assert "UNRELATED_KEY=keep" in contents
    assert "TAVILY_API_KEY=keep-tavily" in contents
    assert "openai-key-123" not in output
    assert "exa-key-123" not in output


def test_setup_wizard_writes_tavily_path(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("EXA_API_KEY=keep-exa\n", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setattr(env, "_BOOTSTRAPPED_DOTENV", None)
    prompts = iter(["tavily", "n"])
    secrets = iter(["openai-tavily", "tavily-secret"])
    preflight_calls: list[str | None] = []

    monkeypatch.setattr("builtins.input", lambda _="": next(prompts))
    monkeypatch.setattr(cli, "getpass", lambda _="": next(secrets))
    monkeypatch.setattr(cli, "print_preflight", lambda project_name=None: preflight_calls.append(project_name) or 0)

    result = cli.run_setup_wizard()

    contents = dotenv_path.read_text(encoding="utf-8")
    assert result == 0
    assert preflight_calls == [None]
    assert "OPENAI_API_KEY=openai-tavily" in contents
    assert "SEARCH_PROVIDER=tavily" in contents
    assert "TAVILY_API_KEY=tavily-secret" in contents
    assert "LANGCHAIN_TRACING_V2=false" in contents
    assert "EXA_API_KEY=keep-exa" in contents


def test_setup_wizard_langsmith_path_defaults_project_and_can_open_browser(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setattr(env, "_BOOTSTRAPPED_DOTENV", None)
    prompts = iter(["none", "y", "", "y"])
    secrets = iter(["openai-langsmith", "langsmith-secret"])
    preflight_calls: list[str | None] = []
    browser_calls: list[str] = []

    monkeypatch.setattr("builtins.input", lambda _="": next(prompts))
    monkeypatch.setattr(cli, "getpass", lambda _="": next(secrets))
    monkeypatch.setattr(cli, "print_preflight", lambda project_name=None: preflight_calls.append(project_name) or 0)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: browser_calls.append(url) or True)

    result = cli.run_setup_wizard()

    contents = dotenv_path.read_text(encoding="utf-8")
    assert result == 0
    assert preflight_calls == ["deepresearch-local"]
    assert browser_calls == ["https://smith.langchain.com/"]
    assert "OPENAI_API_KEY=openai-langsmith" in contents
    assert "SEARCH_PROVIDER=none" in contents
    assert "LANGCHAIN_TRACING_V2=true" in contents
    assert "LANGCHAIN_API_KEY=langsmith-secret" in contents
    assert "LANGCHAIN_PROJECT=deepresearch-local" in contents
