import os
import sys
import types

import pytest

from deepresearch import env


def test_bootstrap_env_loads_dotenv_without_overriding_existing_values(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("OPENAI_API_KEY", "from-process")

    env.bootstrap_env(override=False)

    assert os.environ["OPENAI_API_KEY"] == "from-process"


def test_ensure_runtime_env_ready_raises_actionable_error_when_openai_key_missing(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        env.ensure_runtime_env_ready()


def test_verify_langsmith_auth_returns_clear_missing_key_message(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    ok, message = env.verify_langsmith_auth(project_name="deepresearch")

    assert ok is False
    assert "LANGSMITH_API_KEY" in message


def test_runtime_preflight_reports_required_runtime_key_failure(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    ok, checks = env.runtime_preflight(project_name="deepresearch")

    assert ok is False
    by_name = {check.name: check for check in checks}
    assert by_name["dotenv_file"].ok is True
    assert by_name["runtime_keys"].ok is False
    assert "OPENAI_API_KEY" in by_name["runtime_keys"].message


def test_verify_langsmith_auth_accepts_langsmith_api_key_alias(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    class _FakeClient:
        def list_runs(self, project_name, limit):
            del project_name, limit
            return []

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "key-from-langsmith-var")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "langsmith", types.SimpleNamespace(Client=_FakeClient))

    ok, message = env.verify_langsmith_auth(project_name="deepresearch")

    assert ok is True
    assert "auth OK" in message
