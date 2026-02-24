"""Tests for the online evaluation framework."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from deepresearch import cli, config
from deepresearch.evals.evaluators import (
    _build_process_summary,
    _extract_score,
    _get_final_report,
    _get_user_query,
    eval_answer_quality,
    eval_composite,
    eval_process_quality,
)
from deepresearch.evals.callback import (
    OnlineEvalCallbackHandler,
    attach_online_eval_callback,
    build_eval_callback,
)


# --- Config tests ---


def test_online_evals_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("ENABLE_ONLINE_EVALS", raising=False)
    assert config.online_evals_enabled() is False


def test_online_evals_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("ENABLE_ONLINE_EVALS", "true")
    assert config.online_evals_enabled() is True
    monkeypatch.setenv("ENABLE_ONLINE_EVALS", "false")
    assert config.online_evals_enabled() is False


def test_get_eval_model_defaults(monkeypatch):
    monkeypatch.delenv("EVAL_MODEL", raising=False)
    assert config.get_eval_model() == "openai:gpt-4.1-mini"


def test_get_eval_model_reads_env(monkeypatch):
    monkeypatch.setenv("EVAL_MODEL", "openai:gpt-4o")
    assert config.get_eval_model() == "openai:gpt-4o"


# --- Score extraction ---


def test_extract_score_from_final_line():
    assert _extract_score("Some reasoning here.\n0.85") == 0.85


def test_extract_score_from_inline():
    assert _extract_score("The final score is 0.72 based on analysis.") == 0.72


def test_extract_score_returns_none_for_no_match():
    assert _extract_score("No numeric score here.") is None


def test_extract_score_clamps_to_range():
    assert _extract_score("Score: 0.0") == 0.0
    assert _extract_score("Score: 1.0") == 1.0


# --- Run data extraction ---


def _make_run(inputs=None, outputs=None, trace_id="trace-1", run_id="run-1"):
    return SimpleNamespace(
        id=run_id,
        trace_id=trace_id,
        inputs=inputs or {},
        outputs=outputs or {},
    )


def test_get_final_report_from_final_report_field():
    run = _make_run(outputs={"final_report": "This is the report."})
    assert _get_final_report(run) == "This is the report."


def test_get_final_report_from_messages_fallback():
    run = _make_run(outputs={
        "messages": [
            {"type": "human", "content": "query"},
            {"type": "ai", "content": "The AI response."},
        ]
    })
    assert _get_final_report(run) == "The AI response."


def test_get_final_report_from_messages_with_blocks():
    run = _make_run(outputs={
        "messages": [
            {"type": "ai", "content": [{"type": "text", "text": "block content"}]},
        ]
    })
    assert _get_final_report(run) == "block content"


def test_get_final_report_returns_empty_when_missing():
    run = _make_run(outputs={})
    assert _get_final_report(run) == ""


def test_get_user_query_from_dict_messages():
    run = _make_run(inputs={"messages": [{"type": "human", "content": "What is CRISPR?"}]})
    assert _get_user_query(run) == "What is CRISPR?"


def test_get_user_query_from_object_messages():
    msg = SimpleNamespace(type="human", content="My query")
    run = _make_run(inputs={"messages": [msg]})
    assert _get_user_query(run) == "My query"


def test_get_user_query_returns_empty_when_missing():
    run = _make_run(inputs={})
    assert _get_user_query(run) == ""


# --- Process summary ---


def test_build_process_summary_counts_tools():
    child_runs = [
        SimpleNamespace(name="ConductResearch", inputs={}, outputs={"output": "findings"}),
        SimpleNamespace(name="ConductResearch", inputs={}, outputs={"output": "more findings"}),
        SimpleNamespace(name="search_web", inputs={}, outputs=[
            {"url": "https://example.com/page1"},
            {"url": "https://arxiv.org/paper"},
        ]),
        SimpleNamespace(name="search_web", inputs={}, outputs=[
            {"url": "https://example.com/page2"},
        ]),
        SimpleNamespace(name="fetch_url", inputs={"url": "https://nature.com/article"}, outputs={}),
        SimpleNamespace(name="think_tool", inputs={}, outputs={}),
        SimpleNamespace(name="think_tool", inputs={}, outputs={}),
    ]
    summary = _build_process_summary(child_runs)
    assert "ConductResearch units dispatched: 2" in summary
    assert "search_web calls: 2" in summary
    assert "fetch_url calls: 1" in summary
    assert "think_tool calls: 2" in summary
    assert "arxiv.org" in summary
    assert "example.com" in summary
    assert "nature.com" in summary


def test_build_process_summary_skips_failed_research():
    child_runs = [
        SimpleNamespace(name="ConductResearch", inputs={}, outputs={"output": "[ConductResearch skipped: budget]"}),
        SimpleNamespace(name="ConductResearch", inputs={}, outputs={"output": "real findings"}),
    ]
    summary = _build_process_summary(child_runs)
    assert "ConductResearch units dispatched: 1" in summary


def test_build_process_summary_handles_string_outputs():
    child_runs = [
        SimpleNamespace(name="search_web", inputs={}, outputs="https://example.com/result some text"),
    ]
    summary = _build_process_summary(child_runs)
    assert "search_web calls: 1" in summary
    assert "example.com" in summary


# --- Evaluators with mocked judge ---


def test_eval_answer_quality_returns_zero_when_no_report():
    run = _make_run(outputs={})
    result = eval_answer_quality(run)
    assert result["key"] == "answer_quality"
    assert result["score"] == 0.0
    assert "No final report" in result["comment"]


@patch("deepresearch.evals.evaluators._run_judge")
def test_eval_answer_quality_calls_judge(mock_judge):
    mock_judge.return_value = (0.85, "Good report with citations.")
    run = _make_run(
        inputs={"messages": [{"type": "human", "content": "What is CRISPR?"}]},
        outputs={"final_report": "CRISPR is a gene editing tool [1].\n\nSources:\n[1] https://example.com"},
    )
    result = eval_answer_quality(run)
    assert result["key"] == "answer_quality"
    assert result["score"] == 0.85
    assert mock_judge.called


@patch("deepresearch.evals.evaluators._run_judge")
def test_eval_process_quality_calls_judge(mock_judge):
    mock_judge.return_value = (0.70, "Good search strategy.")
    client = MagicMock()
    client.list_runs.return_value = [
        SimpleNamespace(name="ConductResearch", inputs={}, outputs={"output": "findings"}),
        SimpleNamespace(name="search_web", inputs={}, outputs=[]),
        SimpleNamespace(name="think_tool", inputs={}, outputs={}),
    ]
    run = _make_run()
    result = eval_process_quality(run, client)
    assert result["key"] == "process_quality"
    assert result["score"] == 0.70
    client.list_runs.assert_called_once()


@patch("deepresearch.evals.evaluators._run_judge")
def test_eval_composite_combines_scores(mock_judge):
    mock_judge.side_effect = [
        (0.80, "answer reasoning"),
        (0.60, "process reasoning"),
    ]
    client = MagicMock()
    client.list_runs.return_value = []
    run = _make_run(
        inputs={"messages": [{"type": "human", "content": "query"}]},
        outputs={"final_report": "A report."},
    )
    result = eval_composite(run, client)
    assert result["key"] == "composite_quality"
    expected = round(0.6 * 0.80 + 0.4 * 0.60, 4)
    assert result["score"] == expected
    assert result["answer_result"]["score"] == 0.80
    assert result["process_result"]["score"] == 0.60


def test_eval_process_quality_handles_client_failure():
    client = MagicMock()
    client.list_runs.side_effect = Exception("Network error")
    run = _make_run()
    result = eval_process_quality(run, client)
    assert result["key"] == "process_quality"
    assert result["score"] is None
    assert "Failed to fetch child runs" in result["comment"]


# --- Callback handler ---


def test_build_eval_callback_returns_handler():
    handler = build_eval_callback()
    assert isinstance(handler, OnlineEvalCallbackHandler)


def test_attach_online_eval_callback_appends_handler():
    existing = object()
    cfg = attach_online_eval_callback({"configurable": {"thread_id": "t-1"}, "callbacks": [existing]})
    assert cfg["callbacks"][0] is existing
    assert len(cfg["callbacks"]) == 2
    assert isinstance(cfg["callbacks"][1], OnlineEvalCallbackHandler)


def test_attach_online_eval_callback_dedupes_existing_handler():
    handler = OnlineEvalCallbackHandler(client=MagicMock())
    cfg = attach_online_eval_callback({"callbacks": [handler]})
    assert cfg["callbacks"] == [handler]


def test_callback_handler_skips_non_root_runs():
    handler = OnlineEvalCallbackHandler(client=MagicMock())
    with patch.object(handler, "_run_eval_sync") as mock_eval:
        handler.on_chain_end({}, run_id="run-1", parent_run_id="parent-1")
        mock_eval.assert_not_called()


def test_callback_handler_fires_for_root_runs():
    handler = OnlineEvalCallbackHandler(client=MagicMock())
    with patch.object(handler, "_run_eval_sync") as mock_eval:
        handler.on_chain_end({}, run_id="run-1", parent_run_id=None)
        # Give the thread a moment to start
        import time
        time.sleep(0.1)
        mock_eval.assert_called_once_with("run-1")


def test_callback_handler_skips_when_no_run_id():
    handler = OnlineEvalCallbackHandler(client=MagicMock())
    with patch.object(handler, "_run_eval_sync") as mock_eval:
        handler.on_chain_end({}, run_id=None)
        mock_eval.assert_not_called()


# --- CLI integration ---


def test_cli_thread_config_excludes_callback_when_evals_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_ONLINE_EVALS", "false")
    cfg = cli._thread_config("test-thread")
    assert "callbacks" not in cfg
    assert cfg["configurable"]["thread_id"] == "test-thread"


def test_cli_thread_config_includes_callback_when_evals_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_ONLINE_EVALS", "true")
    cfg = cli._thread_config("test-thread")
    assert "callbacks" in cfg
    assert len(cfg["callbacks"]) == 1
    assert isinstance(cfg["callbacks"][0], OnlineEvalCallbackHandler)


def test_cli_run_passes_callback_when_evals_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_ONLINE_EVALS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")

    fake_app = SimpleNamespace(ainvoke=AsyncMock(return_value={"messages": []}))
    monkeypatch.setattr(cli, "_get_app", lambda: fake_app)

    asyncio.run(cli.run("test query", thread_id="thread-eval"))
    config_payload = fake_app.ainvoke.await_args.kwargs["config"]
    assert "callbacks" in config_payload
    assert len(config_payload["callbacks"]) == 1
