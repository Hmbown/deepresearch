"""
Provider configuration — LLMs and search tools.
"""

from __future__ import annotations

import os
from typing import Literal, cast

from langchain.chat_models import init_chat_model


DEFAULT_ORCHESTRATOR_MODEL = "openai:gpt-5.2"
DEFAULT_SUBAGENT_MODEL = "openai:gpt-5.2"
DEFAULT_SEARCH_PROVIDER = "exa"
SUPPORTED_SEARCH_PROVIDERS = ("exa", "none")
DEFAULT_MAX_STRUCTURED_OUTPUT_RETRIES = 3
DEFAULT_RESEARCHER_SIMPLE_SEARCH_BUDGET = 3
DEFAULT_RESEARCHER_COMPLEX_SEARCH_BUDGET = 5
DEFAULT_MAX_REACT_TOOL_CALLS = 6
DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS = 4
DEFAULT_MAX_RESEARCHER_ITERATIONS = 6
DEFAULT_SUPERVISOR_NOTES_MAX_BULLETS = 10
DEFAULT_SUPERVISOR_NOTES_WORD_BUDGET = 250
DEFAULT_SUPERVISOR_FINAL_REPORT_MAX_SECTIONS = 8
DEFAULT_ENABLE_RUNTIME_EVENT_LOGS = False


class SearchProviderConfigError(RuntimeError):
    """Raised when search provider configuration is invalid or unavailable."""


def _resolve_int_env(var_name: str, default: int, minimum: int = 1) -> int:
    """Resolve an integer config value from env with a safe fallback."""
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default

    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default

    if value < minimum:
        return default
    return value


def _resolve_bool_env(var_name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def get_max_structured_output_retries() -> int:
    """Configured number of structured-output retries (minimum 1)."""
    return _resolve_int_env("MAX_STRUCTURED_OUTPUT_RETRIES", DEFAULT_MAX_STRUCTURED_OUTPUT_RETRIES, minimum=1)


def get_researcher_simple_search_budget() -> int:
    """Configured soft budget for simple researcher search calls."""
    return _resolve_int_env("RESEARCHER_SIMPLE_SEARCH_BUDGET", DEFAULT_RESEARCHER_SIMPLE_SEARCH_BUDGET, minimum=1)


def get_researcher_complex_search_budget() -> int:
    """Configured soft budget for complex researcher search calls."""
    return _resolve_int_env("RESEARCHER_COMPLEX_SEARCH_BUDGET", DEFAULT_RESEARCHER_COMPLEX_SEARCH_BUDGET, minimum=1)


def get_max_react_tool_calls() -> int:
    """Hard cap for researcher ReAct tool calls in one delegation."""
    return _resolve_int_env("MAX_REACT_TOOL_CALLS", DEFAULT_MAX_REACT_TOOL_CALLS, minimum=1)


def get_max_concurrent_research_units() -> int:
    """Hard cap for parallel ConductResearch units per supervisor tool step."""
    return _resolve_int_env(
        "MAX_CONCURRENT_RESEARCH_UNITS",
        DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS,
        minimum=1,
    )


def get_max_researcher_iterations() -> int:
    """Hard cap for total ConductResearch units dispatched in one supervisor run."""
    return _resolve_int_env("MAX_RESEARCHER_ITERATIONS", DEFAULT_MAX_RESEARCHER_ITERATIONS, minimum=1)


def get_supervisor_notes_max_bullets() -> int:
    """Configured cap for compressed supervisor note bullets."""
    return _resolve_int_env("SUPERVISOR_NOTES_MAX_BULLETS", DEFAULT_SUPERVISOR_NOTES_MAX_BULLETS, minimum=1)


def get_supervisor_notes_word_budget() -> int:
    """Configured cap for compressed supervisor note word budget."""
    return _resolve_int_env("SUPERVISOR_NOTES_WORD_BUDGET", DEFAULT_SUPERVISOR_NOTES_WORD_BUDGET, minimum=50)


def get_supervisor_final_report_max_sections() -> int:
    """Configured maximum section count hint for final report drafting."""
    return _resolve_int_env(
        "SUPERVISOR_FINAL_REPORT_MAX_SECTIONS",
        DEFAULT_SUPERVISOR_FINAL_REPORT_MAX_SECTIONS,
        minimum=1,
    )


def _resolve_model_for_role(role: str) -> str:
    """Resolve model string for orchestrator vs subagent roles."""
    if role == "orchestrator":
        return os.environ.get("ORCHESTRATOR_MODEL", DEFAULT_ORCHESTRATOR_MODEL)
    return os.environ.get("SUBAGENT_MODEL", DEFAULT_SUBAGENT_MODEL)


def get_search_provider() -> Literal["exa", "none"]:
    provider = str(os.environ.get("SEARCH_PROVIDER", DEFAULT_SEARCH_PROVIDER)).strip().lower()
    if provider in SUPPORTED_SEARCH_PROVIDERS:
        return cast(Literal["exa", "none"], provider)
    supported = ", ".join(SUPPORTED_SEARCH_PROVIDERS)
    raise SearchProviderConfigError(
        f"Invalid SEARCH_PROVIDER={provider!r}. Supported values: {supported}. "
        "Set SEARCH_PROVIDER=exa (default) or SEARCH_PROVIDER=none."
    )


def runtime_event_logs_enabled() -> bool:
    """Return whether low-noise runtime event logs are enabled."""
    return _resolve_bool_env("ENABLE_RUNTIME_EVENT_LOGS", DEFAULT_ENABLE_RUNTIME_EVENT_LOGS)


def _resolve_required_exa_key() -> str:
    exa_key = str(os.environ.get("EXA_API_KEY", "")).strip()
    if exa_key:
        return exa_key
    raise SearchProviderConfigError(
        "SEARCH_PROVIDER is set to 'exa' but EXA_API_KEY is missing. "
        "Set EXA_API_KEY in .env, or set SEARCH_PROVIDER=none to disable web search explicitly."
    )


def _load_exa_search_results_class():
    try:
        from langchain_exa import ExaSearchResults
    except ImportError as exc:
        raise SearchProviderConfigError(
            "SEARCH_PROVIDER is set to 'exa' but dependency 'langchain-exa' is unavailable. "
            "Install project dependencies (`pip install -e .`) or set SEARCH_PROVIDER=none."
        ) from exc
    return ExaSearchResults


def validate_search_provider_configuration() -> str:
    """Validate configured search provider and return readiness guidance."""
    provider = get_search_provider()
    if provider == "none":
        return "Search provider disabled (`SEARCH_PROVIDER=none`)."

    _resolve_required_exa_key()
    _load_exa_search_results_class()
    return "Search provider ready (`SEARCH_PROVIDER=exa`)."


def get_llm(role: str = "orchestrator"):
    """Return a ChatModel for 'orchestrator' or 'subagent' role.

    Controlled by ORCHESTRATOR_MODEL and SUBAGENT_MODEL.
    Uses init_chat_model for provider detection (e.g. "openai:gpt-5.2").
    """
    model = _resolve_model_for_role(role)
    return init_chat_model(model=model)


def get_search_tool():
    """
    Return a configured web search tool.
    """
    provider = get_search_provider()
    if provider == "none":
        return None

    exa_key = _resolve_required_exa_key()
    exa_search_cls = _load_exa_search_results_class()
    # ExaSearchResults is configured primarily at invocation time.
    return exa_search_cls(exa_api_key=exa_key)
