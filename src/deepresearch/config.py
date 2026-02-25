"""
Provider configuration — LLMs and search tools.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal, cast
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model


DEFAULT_ORCHESTRATOR_MODEL = "openai:gpt-5.2"
DEFAULT_SUBAGENT_MODEL = "openai:gpt-5.2"
DEFAULT_SEARCH_PROVIDER = "openai"
SUPPORTED_SEARCH_PROVIDERS = ("openai", "exa", "tavily", "none")
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
DEFAULT_OPENAI_WEB_SEARCH_MODEL = "gpt-5"
DEFAULT_OPENAI_WEB_SEARCH_CONTEXT_SIZE = "medium"
SUPPORTED_OPENAI_WEB_SEARCH_CONTEXT_SIZES = ("low", "medium", "high")
ModelRole = Literal["orchestrator", "subagent"]


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


def _resolve_model_for_role(role: ModelRole) -> str:
    """Resolve model string for orchestrator vs subagent roles."""
    if role == "orchestrator":
        return os.environ.get("ORCHESTRATOR_MODEL", DEFAULT_ORCHESTRATOR_MODEL)
    if role == "subagent":
        return os.environ.get("SUBAGENT_MODEL", DEFAULT_SUBAGENT_MODEL)
    raise ValueError(f"Unsupported model role: {role!r}")


def get_search_provider() -> Literal["openai", "exa", "tavily", "none"]:
    provider = str(os.environ.get("SEARCH_PROVIDER", DEFAULT_SEARCH_PROVIDER)).strip().lower()
    if provider in SUPPORTED_SEARCH_PROVIDERS:
        return cast(Literal["openai", "exa", "tavily", "none"], provider)
    supported = ", ".join(SUPPORTED_SEARCH_PROVIDERS)
    raise SearchProviderConfigError(
        f"Invalid SEARCH_PROVIDER={provider!r}. Supported values: {supported}. "
        "Set SEARCH_PROVIDER=openai (default), SEARCH_PROVIDER=tavily, "
        "SEARCH_PROVIDER=exa, or SEARCH_PROVIDER=none."
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


def _resolve_required_openai_key() -> str:
    openai_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if openai_key:
        return openai_key
    raise SearchProviderConfigError(
        "SEARCH_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing. "
        "Set OPENAI_API_KEY in .env, or set SEARCH_PROVIDER=none to disable web search explicitly."
    )


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


def _load_tavily_search_class():
    try:
        from langchain_tavily import TavilySearch
    except ImportError as exc:
        raise SearchProviderConfigError(
            "SEARCH_PROVIDER is set to 'tavily' but dependency 'langchain-tavily' is unavailable. "
            "Install project dependencies (`pip install -e .`) or set SEARCH_PROVIDER=none."
        ) from exc
    return TavilySearch


def _load_openai_client_class():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SearchProviderConfigError(
            "SEARCH_PROVIDER is set to 'openai' but dependency 'openai' is unavailable. "
            "Install project dependencies (`pip install -e .`) or set SEARCH_PROVIDER=none."
        ) from exc
    return OpenAI


def _normalize_openai_search_model(raw_model: str) -> str:
    model = str(raw_model or "").strip()
    if model.startswith("openai:"):
        model = model.split(":", 1)[1].strip()
    return model


def _resolve_openai_web_search_model() -> str:
    configured_model = _normalize_openai_search_model(os.environ.get("OPENAI_WEB_SEARCH_MODEL", ""))
    if configured_model:
        return configured_model
    role_model = _normalize_openai_search_model(_resolve_model_for_role("subagent"))
    return role_model or DEFAULT_OPENAI_WEB_SEARCH_MODEL


def _resolve_openai_web_search_context_size() -> str:
    raw_value = str(
        os.environ.get("OPENAI_WEB_SEARCH_CONTEXT_SIZE", DEFAULT_OPENAI_WEB_SEARCH_CONTEXT_SIZE)
    ).strip().lower()
    if raw_value in SUPPORTED_OPENAI_WEB_SEARCH_CONTEXT_SIZES:
        return raw_value
    return DEFAULT_OPENAI_WEB_SEARCH_CONTEXT_SIZE


def _resolve_openai_web_search_allowed_domains() -> list[str]:
    raw_domains = str(os.environ.get("OPENAI_WEB_SEARCH_ALLOWED_DOMAINS", "")).strip()
    if not raw_domains:
        return []

    normalized_domains: list[str] = []
    for item in raw_domains.split(","):
        domain = str(item).strip().lower()
        if not domain:
            continue
        parsed = urlparse(domain if "://" in domain else f"https://{domain}")
        hostname = (parsed.netloc or parsed.path).strip().lower().rstrip("/")
        if hostname and hostname not in normalized_domains:
            normalized_domains.append(hostname)
    return normalized_domains


def _resolve_openai_web_search_user_location() -> dict[str, str] | None:
    city = str(os.environ.get("OPENAI_WEB_SEARCH_USER_CITY", "")).strip()
    country = str(os.environ.get("OPENAI_WEB_SEARCH_USER_COUNTRY", "")).strip()
    region = str(os.environ.get("OPENAI_WEB_SEARCH_USER_REGION", "")).strip()
    timezone = str(os.environ.get("OPENAI_WEB_SEARCH_USER_TIMEZONE", "")).strip()

    if not any([city, country, region, timezone]):
        return None

    payload: dict[str, str] = {"type": "approximate"}
    if city:
        payload["city"] = city
    if country:
        payload["country"] = country
    if region:
        payload["region"] = region
    if timezone:
        payload["timezone"] = timezone
    return payload


class _OpenAIResponsesWebSearchTool:
    """Adapter for OpenAI Responses API built-in web_search tool."""

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        context_size: str,
        allowed_domains: list[str] | None = None,
        user_location: dict[str, str] | None = None,
    ) -> None:
        self._client = _load_openai_client_class()(api_key=api_key)
        self._model = model
        self._context_size = context_size
        self._allowed_domains = list(allowed_domains or [])
        self._user_location = dict(user_location or {})

    def _resolve_query(self, payload: Any) -> str:
        if isinstance(payload, dict):
            query = str(payload.get("query", "")).strip()
            if query:
                return query
        return str(payload or "").strip()

    def _tool_definition(self) -> dict[str, Any]:
        tool_payload: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": self._context_size,
        }
        if self._allowed_domains:
            tool_payload["filters"] = {"allowed_domains": self._allowed_domains}
        if self._user_location:
            tool_payload["user_location"] = self._user_location
        return tool_payload

    @staticmethod
    def _field(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _citation_metadata(self, output_items: list[Any]) -> dict[str, tuple[str, str]]:
        citation_map: dict[str, tuple[str, str]] = {}
        for item in output_items:
            if self._field(item, "type", "") != "message":
                continue
            content_items = self._field(item, "content", [])
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                if self._field(content_item, "type", "") != "output_text":
                    continue
                text = str(self._field(content_item, "text", "") or "")
                annotations = self._field(content_item, "annotations", [])
                if not isinstance(annotations, list):
                    continue
                for annotation in annotations:
                    if self._field(annotation, "type", "") != "url_citation":
                        continue
                    url = str(self._field(annotation, "url", "") or "").strip()
                    if not url:
                        continue
                    title = str(self._field(annotation, "title", "") or "").strip()
                    start_idx = int(self._field(annotation, "start_index", 0) or 0)
                    end_idx = int(self._field(annotation, "end_index", 0) or 0)
                    if 0 <= start_idx < end_idx <= len(text):
                        snippet = text[max(0, start_idx - 120) : min(len(text), end_idx + 120)].strip()
                    else:
                        snippet = ""
                    citation_map[url] = (title, snippet)
        return citation_map

    def _extract_results(self, response: Any) -> list[dict[str, str]]:
        output_items = self._field(response, "output", [])
        if not isinstance(output_items, list):
            return []

        citation_map = self._citation_metadata(output_items)
        ordered_urls: list[str] = []

        for item in output_items:
            if self._field(item, "type", "") != "web_search_call":
                continue
            action = self._field(item, "action", None)
            if action is None or self._field(action, "type", "") != "search":
                continue
            sources = self._field(action, "sources", [])
            if not isinstance(sources, list):
                continue
            for source in sources:
                if self._field(source, "type", "") != "url":
                    continue
                url = str(self._field(source, "url", "") or "").strip()
                if url and url not in ordered_urls:
                    ordered_urls.append(url)

        for cited_url in citation_map:
            if cited_url and cited_url not in ordered_urls:
                ordered_urls.append(cited_url)

        results: list[dict[str, str]] = []
        for url in ordered_urls:
            title, snippet = citation_map.get(url, ("", ""))
            fallback_title = (urlparse(url).netloc or "Web source").strip()
            results.append(
                {
                    "title": title or fallback_title,
                    "url": url,
                    "content": snippet or title or "",
                }
            )
        return results

    def invoke(self, payload: Any) -> dict[str, Any]:
        query = self._resolve_query(payload)
        if not query:
            return {"results": []}
        response = self._client.responses.create(
            model=self._model,
            input=query,
            tools=[self._tool_definition()],
        )
        return {"results": self._extract_results(response)}

    async def ainvoke(self, payload: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.invoke, payload)


def _build_openai_responses_web_search_tool() -> _OpenAIResponsesWebSearchTool:
    api_key = _resolve_required_openai_key()
    _load_openai_client_class()
    return _OpenAIResponsesWebSearchTool(
        api_key=api_key,
        model=_resolve_openai_web_search_model(),
        context_size=_resolve_openai_web_search_context_size(),
        allowed_domains=_resolve_openai_web_search_allowed_domains(),
        user_location=_resolve_openai_web_search_user_location(),
    )


def validate_search_provider_configuration() -> str:
    """Validate configured search provider and return readiness guidance."""
    provider = get_search_provider()
    if provider == "none":
        return "Search provider disabled (`SEARCH_PROVIDER=none`)."

    if provider == "openai":
        _resolve_required_openai_key()
        _load_openai_client_class()
        return "Search provider ready (`SEARCH_PROVIDER=openai`)."

    if provider == "exa":
        _resolve_required_exa_key()
        _load_exa_search_results_class()
        return "Search provider ready (`SEARCH_PROVIDER=exa`)."

    _resolve_required_tavily_key()
    _load_tavily_search_class()
    return "Search provider ready (`SEARCH_PROVIDER=tavily`)."


def get_model_string(role: ModelRole = "orchestrator") -> str:
    """Return the raw provider:model string (e.g. ``"openai:gpt-5.2"``).

    Callers that need a pre-configured ChatModel with Responses API flags
    should use ``get_llm()`` instead.
    """
    return _resolve_model_for_role(role)


def get_llm(role: ModelRole = "orchestrator", *, prefer_compact_context: bool = False):
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
        if openai_use_previous_response_id_enabled() or prefer_compact_context:
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

    if provider == "openai":
        return _build_openai_responses_web_search_tool()

    if provider == "exa":
        exa_key = _resolve_required_exa_key()
        exa_search_cls = _load_exa_search_results_class()
        # ExaSearchResults is configured primarily at invocation time.
        return exa_search_cls(exa_api_key=exa_key)

    _resolve_required_tavily_key()  # validate key exists before constructing
    tavily_search_cls = _load_tavily_search_class()
    return tavily_search_cls()  # reads TAVILY_API_KEY from env
