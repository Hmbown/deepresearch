"""
Provider configuration — LLMs and search tools.
"""

from __future__ import annotations

import os
from typing import Any, Literal, cast

from langchain.chat_models import init_chat_model


DEFAULT_ORCHESTRATOR_MODEL = "openai:gpt-5.2"
DEFAULT_SUBAGENT_MODEL = "openai:gpt-5.2"
DEFAULT_SEARCH_PROVIDER = "exa"
SUPPORTED_SEARCH_PROVIDERS = ("exa", "tavily", "none")
DEFAULT_MAX_STRUCTURED_OUTPUT_RETRIES = 3
DEFAULT_MAX_REACT_TOOL_CALLS = 40
DEFAULT_MAX_CONCURRENT_RESEARCH_UNITS = 4
DEFAULT_MAX_RESEARCHER_ITERATIONS = 60
DEFAULT_ENABLE_RUNTIME_EVENT_LOGS = False
DEFAULT_EVAL_MODEL = "openai:gpt-4.1-mini"
DEFAULT_ENABLE_ONLINE_EVALS = False
DEFAULT_OPENAI_USE_RESPONSES_API = True
DEFAULT_OPENAI_USE_PREVIOUS_RESPONSE_ID = False
DEFAULT_OPENAI_OUTPUT_VERSION = "responses/v1"


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


def _resolve_model_for_role(role: str) -> str:
    """Resolve model string for orchestrator vs subagent roles."""
    if role == "orchestrator":
        return os.environ.get("ORCHESTRATOR_MODEL", DEFAULT_ORCHESTRATOR_MODEL)
    return os.environ.get("SUBAGENT_MODEL", DEFAULT_SUBAGENT_MODEL)


def get_search_provider() -> Literal["exa", "tavily", "none"]:
    provider = str(os.environ.get("SEARCH_PROVIDER", DEFAULT_SEARCH_PROVIDER)).strip().lower()
    if provider in SUPPORTED_SEARCH_PROVIDERS:
        return cast(Literal["exa", "tavily", "none"], provider)
    supported = ", ".join(SUPPORTED_SEARCH_PROVIDERS)
    raise SearchProviderConfigError(
        f"Invalid SEARCH_PROVIDER={provider!r}. Supported values: {supported}. "
        "Set SEARCH_PROVIDER=exa (default), SEARCH_PROVIDER=tavily, or SEARCH_PROVIDER=none."
    )


def runtime_event_logs_enabled() -> bool:
    """Return whether low-noise runtime event logs are enabled."""
    return _resolve_bool_env("ENABLE_RUNTIME_EVENT_LOGS", DEFAULT_ENABLE_RUNTIME_EVENT_LOGS)


def online_evals_enabled() -> bool:
    """Return whether online LLM-as-judge evaluations are enabled."""
    return _resolve_bool_env("ENABLE_ONLINE_EVALS", DEFAULT_ENABLE_ONLINE_EVALS)


def openai_responses_api_enabled() -> bool:
    """Return whether OpenAI models should use the Responses API."""
    return _resolve_bool_env("OPENAI_USE_RESPONSES_API", DEFAULT_OPENAI_USE_RESPONSES_API)


def openai_use_previous_response_id_enabled() -> bool:
    """Return whether OpenAI Responses API should use previous_response_id compaction."""
    return _resolve_bool_env(
        "OPENAI_USE_PREVIOUS_RESPONSE_ID",
        DEFAULT_OPENAI_USE_PREVIOUS_RESPONSE_ID,
    )


def get_openai_output_version() -> str | None:
    """Return output version for OpenAI chat models, if configured."""
    output_version = str(os.environ.get("OPENAI_OUTPUT_VERSION", DEFAULT_OPENAI_OUTPUT_VERSION)).strip()
    return output_version or None


def get_eval_model() -> str:
    """Return the model string for eval judge (e.g. ``"openai:gpt-4.1-mini"``)."""
    return os.environ.get("EVAL_MODEL", DEFAULT_EVAL_MODEL)


def _resolve_required_exa_key() -> str:
    exa_key = str(os.environ.get("EXA_API_KEY", "")).strip()
    if exa_key:
        return exa_key
    raise SearchProviderConfigError(
        "SEARCH_PROVIDER is set to 'exa' but EXA_API_KEY is missing. "
        "Set EXA_API_KEY in .env, or set SEARCH_PROVIDER=none to disable web search explicitly."
    )


def _resolve_required_tavily_key() -> str:
    tavily_key = str(os.environ.get("TAVILY_API_KEY", "")).strip()
    if tavily_key:
        return tavily_key
    raise SearchProviderConfigError(
        "SEARCH_PROVIDER is set to 'tavily' but TAVILY_API_KEY is missing. "
        "Set TAVILY_API_KEY in .env, or set SEARCH_PROVIDER=none to disable web search explicitly."
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


def _load_tavily_search_results_class():
    try:
        from langchain_tavily import TavilySearchResults
    except ImportError as exc:
        raise SearchProviderConfigError(
            "SEARCH_PROVIDER is set to 'tavily' but dependency 'langchain-tavily' is unavailable. "
            "Install project dependencies (`pip install -e .`) or set SEARCH_PROVIDER=none."
        ) from exc
    return TavilySearchResults


def validate_search_provider_configuration() -> str:
    """Validate configured search provider and return readiness guidance."""
    provider = get_search_provider()
    if provider == "none":
        return "Search provider disabled (`SEARCH_PROVIDER=none`)."

    if provider == "exa":
        _resolve_required_exa_key()
        _load_exa_search_results_class()
        return "Search provider ready (`SEARCH_PROVIDER=exa`)."

    _resolve_required_tavily_key()
    _load_tavily_search_results_class()
    return "Search provider ready (`SEARCH_PROVIDER=tavily`)."


def get_model_string(role: str = "orchestrator") -> str:
    """Return the raw provider:model string (e.g. ``"openai:gpt-5.2"``).

    Useful for callers like ``create_deep_agent()`` that accept model strings
    directly and apply their own provider-specific handling.
    """
    return _resolve_model_for_role(role)


def get_llm(role: str = "orchestrator"):
    """Return a ChatModel for 'orchestrator' or 'subagent' role.

    Controlled by ORCHESTRATOR_MODEL and SUBAGENT_MODEL.
    Uses init_chat_model for provider detection (e.g. "openai:gpt-5.2").
    """
    model = _resolve_model_for_role(role)
    init_kwargs: dict[str, Any] = {"model": model}

    if model.startswith("openai:") and openai_responses_api_enabled():
        init_kwargs["use_responses_api"] = True
        output_version = get_openai_output_version()
        if output_version:
            init_kwargs["output_version"] = output_version
        if openai_use_previous_response_id_enabled():
            init_kwargs["use_previous_response_id"] = True

    try:
        return init_chat_model(**init_kwargs)
    except TypeError:
        if model.startswith("openai:") and "use_responses_api" in init_kwargs:
            return init_chat_model(model=model)
        raise


def get_search_tool():
    """
    Return a configured web search tool.
    """
    provider = get_search_provider()
    if provider == "none":
        return None

    if provider == "exa":
        exa_key = _resolve_required_exa_key()
        exa_search_cls = _load_exa_search_results_class()
        # ExaSearchResults is configured primarily at invocation time.
        return exa_search_cls(exa_api_key=exa_key)

    tavily_key = _resolve_required_tavily_key()
    tavily_search_cls = _load_tavily_search_results_class()
    return tavily_search_cls(api_key=tavily_key)
