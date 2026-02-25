"""Environment bootstrap and preflight checks for local/runtime execution."""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
import sys
from collections.abc import Mapping

from dotenv import load_dotenv

from .config import (
    DEFAULT_ORCHESTRATOR_MODEL,
    DEFAULT_SUBAGENT_MODEL,
    SearchProviderConfigError,
    get_search_provider,
    validate_search_provider_configuration,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DOTENV = _PROJECT_ROOT / ".env"
_DEFAULT_PROJECT_DOTENV = _PROJECT_DOTENV
_ENV_FILE_OVERRIDE_VAR = "DEEPRESEARCH_ENV_FILE"
_ENV_BOOTSTRAPPED = False
_BOOTSTRAPPED_DOTENV: Path | None = None
_ENV_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


@dataclass(frozen=True)
class PreflightCheckResult:
    """Result of one runtime preflight check."""

    name: str
    ok: bool
    message: str


def project_root() -> Path:
    """Return repository root path."""
    return project_dotenv_path().parent


def project_dotenv_path() -> Path:
    """Return expected project .env path."""
    override = os.environ.get(_ENV_FILE_OVERRIDE_VAR, "").strip()
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = (Path.cwd() / override_path).resolve()
        return override_path

    # Preserve compatibility for tests that monkeypatch `_PROJECT_DOTENV`.
    if _PROJECT_DOTENV != _DEFAULT_PROJECT_DOTENV:
        return _PROJECT_DOTENV

    discovered = _discover_dotenv_path_from_cwd()
    if discovered is not None:
        return discovered

    return _PROJECT_DOTENV


def bootstrap_env(*, override: bool = False) -> Path:
    """Load the project .env file once per process."""
    global _ENV_BOOTSTRAPPED, _BOOTSTRAPPED_DOTENV
    dotenv_path = project_dotenv_path()
    should_reload = override or not _ENV_BOOTSTRAPPED or _BOOTSTRAPPED_DOTENV != dotenv_path
    if should_reload:
        load_dotenv(dotenv_path=dotenv_path, override=override)
        _ENV_BOOTSTRAPPED = True
        _BOOTSTRAPPED_DOTENV = dotenv_path
    return dotenv_path


def _serialize_dotenv_value(value: str) -> str:
    raw = str(value)
    if raw == "":
        return '""'
    if any(character.isspace() for character in raw) or "#" in raw:
        escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return raw


def update_project_dotenv(updates: Mapping[str, str]) -> Path:
    """Upsert selected keys in the active project dotenv file."""
    dotenv_path = project_dotenv_path()
    normalized_updates = {str(key): str(value) for key, value in updates.items()}
    if not normalized_updates:
        return dotenv_path

    existing_lines: list[str] = []
    if dotenv_path.exists():
        existing_lines = dotenv_path.read_text(encoding="utf-8").splitlines(keepends=True)

    remaining_keys = set(normalized_updates)
    rewritten_lines: list[str] = []
    for line in existing_lines:
        key_match = _ENV_ASSIGNMENT_RE.match(line)
        if key_match is None:
            rewritten_lines.append(line)
            continue

        key = key_match.group(1)
        if key not in normalized_updates:
            rewritten_lines.append(line)
            continue

        rewritten_lines.append(f"{key}={_serialize_dotenv_value(normalized_updates[key])}\n")
        remaining_keys.discard(key)

    if rewritten_lines and not rewritten_lines[-1].endswith("\n"):
        rewritten_lines[-1] += "\n"

    for key, value in normalized_updates.items():
        if key in remaining_keys:
            rewritten_lines.append(f"{key}={_serialize_dotenv_value(value)}\n")

    dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    dotenv_path.write_text("".join(rewritten_lines), encoding="utf-8")

    # Ensure follow-up checks in the same process observe the new values.
    bootstrap_env(override=True)
    return dotenv_path


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _langsmith_tracing_enabled() -> bool:
    return _is_truthy(os.environ.get("LANGCHAIN_TRACING_V2")) or _is_truthy(os.environ.get("LANGSMITH_TRACING"))


def _langsmith_api_key() -> str | None:
    return os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")


def _langsmith_project_name(default: str) -> str:
    return (
        os.environ.get("LANGCHAIN_PROJECT")
        or os.environ.get("LANGSMITH_PROJECT")
        or default
    )


def _dependency_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _looks_like_deepresearch_repo(pyproject_path: Path) -> bool:
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return 'name = "deepresearch"' in text


def _discover_dotenv_path_from_cwd() -> Path | None:
    try:
        cwd = Path.cwd().resolve()
    except Exception:
        return None

    for directory in [cwd, *cwd.parents]:
        dotenv_candidate = directory / ".env"
        if dotenv_candidate.exists():
            return dotenv_candidate

    for directory in [cwd, *cwd.parents]:
        template_candidate = directory / ".env.example"
        if template_candidate.exists():
            return directory / ".env"

    for directory in [cwd, *cwd.parents]:
        pyproject_path = directory / "pyproject.toml"
        if pyproject_path.exists() and _looks_like_deepresearch_repo(pyproject_path):
            return directory / ".env"

    return None


def _model_requires_openai_key(model: str) -> bool:
    return str(model).strip().lower().startswith("openai:")


def _required_runtime_env_vars() -> set[str]:
    required: set[str] = set()
    orchestrator_model = str(os.environ.get("ORCHESTRATOR_MODEL", DEFAULT_ORCHESTRATOR_MODEL)).strip()
    subagent_model = str(os.environ.get("SUBAGENT_MODEL", DEFAULT_SUBAGENT_MODEL)).strip()
    if _model_requires_openai_key(orchestrator_model) or _model_requires_openai_key(subagent_model):
        required.add("OPENAI_API_KEY")

    try:
        search_provider = get_search_provider()
    except SearchProviderConfigError:
        search_provider = None

    if search_provider == "openai":
        required.add("OPENAI_API_KEY")
    elif search_provider == "exa":
        required.add("EXA_API_KEY")
    elif search_provider == "tavily":
        required.add("TAVILY_API_KEY")
    return required


def missing_runtime_env_vars() -> list[str]:
    """Return required runtime env vars that are currently unset."""
    required = _required_runtime_env_vars()
    return sorted(name for name in required if not str(os.environ.get(name, "")).strip())


def ensure_runtime_env_ready() -> None:
    """Raise a clear error when required runtime env vars are missing."""
    dotenv_path = bootstrap_env(override=False)
    missing = missing_runtime_env_vars()
    if not missing:
        try:
            validate_search_provider_configuration()
        except SearchProviderConfigError as exc:
            raise RuntimeError(str(exc)) from exc
        return
    joined = ", ".join(missing)
    raise RuntimeError(
        "Missing required environment variable(s): "
        f"{joined}. Set them in your shell or add them to `{dotenv_path}`."
    )


def verify_langsmith_auth(project_name: str | None = None) -> tuple[bool, str]:
    """Verify LangSmith auth when tracing is enabled; return status + guidance."""
    bootstrap_env(override=False)
    tracing_enabled = _langsmith_tracing_enabled()
    if not tracing_enabled:
        return (
            True,
            "LangSmith tracing is disabled (`LANGCHAIN_TRACING_V2`/`LANGSMITH_TRACING` not truthy); auth check skipped.",
        )

    api_key = _langsmith_api_key()
    if not api_key:
        return (
            False,
            "LangSmith tracing is enabled but no API key was found. Set `LANGCHAIN_API_KEY` or `LANGSMITH_API_KEY` in `.env`.",
        )

    resolved_project = project_name or _langsmith_project_name("deepresearch-local")
    try:
        from langsmith import Client

        client = Client()
        list(client.list_runs(project_name=resolved_project, limit=1))
    except Exception as exc:  # pragma: no cover - network/provider dependent
        error_head = str(exc).splitlines()[0].strip() or exc.__class__.__name__
        return (
            False,
            "LangSmith auth failed. Verify `LANGCHAIN_API_KEY` and `LANGCHAIN_ENDPOINT`. "
            f"Underlying error: {error_head}",
        )
    return (True, f"LangSmith auth OK for project `{resolved_project}`.")


def runtime_preflight(project_name: str | None = None) -> tuple[bool, list[PreflightCheckResult]]:
    """Run setup checks useful for first-time local setup."""
    dotenv_path = bootstrap_env(override=False)
    checks: list[PreflightCheckResult] = []
    missing = missing_runtime_env_vars()

    if dotenv_path.exists():
        checks.append(PreflightCheckResult("dotenv_file", True, f"Found `{dotenv_path}`."))
    elif not missing:
        checks.append(
            PreflightCheckResult(
                "dotenv_file",
                True,
                f"No `.env` file at `{dotenv_path}`; using environment variables from current shell/process.",
            )
        )
    else:
        template_path = dotenv_path.with_name(".env.example")
        setup_hint = (
            f"Copy `{template_path}` to `{dotenv_path}`."
            if template_path.exists()
            else f"Create `{dotenv_path}` and set required keys."
        )
        checks.append(
            PreflightCheckResult(
                "dotenv_file",
                False,
                f"Missing `{dotenv_path}`. {setup_hint}",
            )
        )

    if missing:
        checks.append(
            PreflightCheckResult(
                "runtime_keys",
                False,
                "Missing required key(s): " + ", ".join(missing),
            )
        )
    else:
        checks.append(PreflightCheckResult("runtime_keys", True, "Required runtime key(s) are set."))

    try:
        search_provider_message = validate_search_provider_configuration()
        checks.append(PreflightCheckResult("search_provider", True, search_provider_message))
    except SearchProviderConfigError as exc:
        checks.append(PreflightCheckResult("search_provider", False, str(exc)))

    if _dependency_available("deepagents"):
        checks.append(PreflightCheckResult("deepagents", True, "deepagents runtime dependency is installed."))
    else:
        checks.append(
            PreflightCheckResult(
                "deepagents",
                False,
                "Missing required runtime dependency `deepagents`. "
                f"Install dependencies with `{sys.executable} -m pip install -e .` "
                "for this interpreter.",
            )
        )

    langsmith_ok, langsmith_message = verify_langsmith_auth(project_name=project_name)
    checks.append(PreflightCheckResult("langsmith", langsmith_ok, langsmith_message))

    overall_ok = all(check.ok for check in checks)
    return overall_ok, checks
