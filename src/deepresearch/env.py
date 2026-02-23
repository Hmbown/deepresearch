"""Environment bootstrap and preflight checks for local/runtime execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .config import SearchProviderConfigError, validate_search_provider_configuration

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DOTENV = _PROJECT_ROOT / ".env"
_ENV_BOOTSTRAPPED = False


@dataclass(frozen=True)
class PreflightCheckResult:
    """Result of one runtime preflight check."""

    name: str
    ok: bool
    message: str


def project_root() -> Path:
    """Return repository root path."""
    return _PROJECT_ROOT


def project_dotenv_path() -> Path:
    """Return expected project .env path."""
    return _PROJECT_DOTENV


def bootstrap_env(*, override: bool = False) -> Path:
    """Load the project .env file once per process."""
    global _ENV_BOOTSTRAPPED
    if not _ENV_BOOTSTRAPPED:
        load_dotenv(dotenv_path=_PROJECT_DOTENV, override=override)
        _ENV_BOOTSTRAPPED = True
    return _PROJECT_DOTENV


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


def missing_runtime_env_vars() -> list[str]:
    """Return required runtime env vars that are currently unset."""
    required = ("OPENAI_API_KEY",)
    return [name for name in required if not os.environ.get(name)]


def ensure_runtime_env_ready() -> None:
    """Raise a clear error when required runtime env vars are missing."""
    bootstrap_env(override=False)
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
        f"{joined}. Copy .env.example to .env and set the missing value(s)."
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

    if dotenv_path.exists():
        checks.append(PreflightCheckResult("dotenv_file", True, f"Found `{dotenv_path}`."))
    else:
        checks.append(
            PreflightCheckResult(
                "dotenv_file",
                False,
                f"Missing `{dotenv_path}`. Copy `.env.example` to `.env`.",
            )
        )

    missing = missing_runtime_env_vars()
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

    langsmith_ok, langsmith_message = verify_langsmith_auth(project_name=project_name)
    checks.append(PreflightCheckResult("langsmith", langsmith_ok, langsmith_message))

    overall_ok = all(check.ok for check in checks)
    return overall_ok, checks
