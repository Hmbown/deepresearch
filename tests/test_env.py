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


def test_ensure_runtime_env_ready_allows_non_openai_models_with_search_disabled(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "anthropic:claude-opus-4-6")
    monkeypatch.setenv("SUBAGENT_MODEL", "anthropic:claude-sonnet-4-5")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env.ensure_runtime_env_ready()


def test_ensure_runtime_env_ready_requires_openai_key_for_openai_search_provider(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "anthropic:claude-opus-4-6")
    monkeypatch.setenv("SUBAGENT_MODEL", "anthropic:claude-sonnet-4-5")
    monkeypatch.setenv("SEARCH_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        env.ensure_runtime_env_ready()


def test_missing_runtime_env_vars_require_only_provider_specific_key_for_non_openai_models(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MODEL", "anthropic:claude-opus-4-6")
    monkeypatch.setenv("SUBAGENT_MODEL", "anthropic:claude-sonnet-4-5")
    monkeypatch.setenv("SEARCH_PROVIDER", "exa")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    assert env.missing_runtime_env_vars() == ["EXA_API_KEY"]


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
    monkeypatch.setenv("SEARCH_PROVIDER", "none")

    ok, checks = env.runtime_preflight(project_name="deepresearch")

    assert ok is False
    by_name = {check.name: check for check in checks}
    assert by_name["dotenv_file"].ok is True
    assert by_name["runtime_keys"].ok is False
    assert "OPENAI_API_KEY" in by_name["runtime_keys"].message


def test_ensure_runtime_env_ready_raises_for_missing_exa_key_when_exa_selected(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SEARCH_PROVIDER", "exa")
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        env.ensure_runtime_env_ready()


def test_runtime_preflight_reports_invalid_search_provider(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SEARCH_PROVIDER", "invalid-provider")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    ok, checks = env.runtime_preflight(project_name="deepresearch")

    assert ok is False
    by_name = {check.name: check for check in checks}
    assert by_name["search_provider"].ok is False
    assert "Invalid SEARCH_PROVIDER" in by_name["search_provider"].message


def test_runtime_preflight_reports_deepagents_dependency(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setattr(env, "missing_runtime_env_vars", lambda: [])
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(env, "_dependency_available", lambda _: False)

    ok, checks = env.runtime_preflight(project_name="deepresearch")

    assert ok is False
    by_name = {check.name: check for check in checks}
    assert by_name["deepagents"].ok is False
    assert "Missing required runtime dependency `deepagents`" in by_name["deepagents"].message


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


def test_project_dotenv_path_uses_explicit_override(monkeypatch, tmp_path):
    override_path = tmp_path / "custom.env"
    monkeypatch.setenv("DEEPRESEARCH_ENV_FILE", str(override_path))

    assert env.project_dotenv_path() == override_path


def test_project_dotenv_path_discovers_cwd_template(monkeypatch, tmp_path):
    project_root = tmp_path / "deepresearch"
    project_root.mkdir()
    (project_root / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    (project_root / "pyproject.toml").write_text('[project]\nname = "deepresearch"\n', encoding="utf-8")

    monkeypatch.delenv("DEEPRESEARCH_ENV_FILE", raising=False)
    monkeypatch.setattr(env, "_PROJECT_DOTENV", env._DEFAULT_PROJECT_DOTENV)
    monkeypatch.chdir(project_root)

    assert env.project_dotenv_path() == project_root / ".env"


def test_runtime_preflight_allows_process_env_without_dotenv(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    monkeypatch.setenv("DEEPRESEARCH_ENV_FILE", str(dotenv_path))
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(env, "_dependency_available", lambda _: True)

    ok, checks = env.runtime_preflight(project_name="deepresearch")

    assert ok is True
    by_name = {check.name: check for check in checks}
    assert by_name["dotenv_file"].ok is True
    assert "using environment variables" in by_name["dotenv_file"].message


def test_update_project_dotenv_upserts_managed_keys_and_preserves_existing_entries(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "# existing comment\n"
        "UNRELATED_KEY=keep-me\n"
        "OPENAI_API_KEY=old-openai\n"
        "export SEARCH_PROVIDER=none\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(env, "_PROJECT_DOTENV", dotenv_path)
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setattr(env, "_BOOTSTRAPPED_DOTENV", None)

    updated_path = env.update_project_dotenv(
        {
            "OPENAI_API_KEY": "new-openai-key",
            "SEARCH_PROVIDER": "exa",
            "EXA_API_KEY": "new-exa-key",
        }
    )

    contents = dotenv_path.read_text(encoding="utf-8")
    assert updated_path == dotenv_path
    assert "# existing comment" in contents
    assert "UNRELATED_KEY=keep-me" in contents
    assert contents.count("OPENAI_API_KEY=") == 1
    assert "OPENAI_API_KEY=new-openai-key" in contents
    assert "SEARCH_PROVIDER=exa" in contents
    assert "EXA_API_KEY=new-exa-key" in contents
    assert os.environ["OPENAI_API_KEY"] == "new-openai-key"


def test_update_project_dotenv_honors_explicit_env_file_override(monkeypatch, tmp_path):
    override_path = tmp_path / "custom.env"
    override_path.write_text("UNRELATED=1", encoding="utf-8")

    monkeypatch.setenv("DEEPRESEARCH_ENV_FILE", str(override_path))
    monkeypatch.setattr(env, "_ENV_BOOTSTRAPPED", False)
    monkeypatch.setattr(env, "_BOOTSTRAPPED_DOTENV", None)

    updated_path = env.update_project_dotenv({"OPENAI_API_KEY": "override-key", "SEARCH_PROVIDER": "none"})

    contents = override_path.read_text(encoding="utf-8")
    assert updated_path == override_path
    assert contents.startswith("UNRELATED=1\n")
    assert "OPENAI_API_KEY=override-key" in contents
    assert "SEARCH_PROVIDER=none" in contents
