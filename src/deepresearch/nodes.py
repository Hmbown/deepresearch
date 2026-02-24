"""Reusable research tools and deterministic search post-processing."""

from __future__ import annotations

from collections.abc import Callable
import ipaddress
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool

MAX_SEARCH_RESULTS_FOR_AGENT = 8


def _is_non_public_ip(hostname: str) -> bool:
    """Return True when hostname is a private/special-use IP literal."""
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_fetch_url_target(url: str) -> tuple[bool, str]:
    """Validate URL target to reduce SSRF risk for non-public hosts."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "URL scheme must be http or https"

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False, "URL must include a hostname"

    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        return False, "target host is not publicly routable"

    if hostname.endswith(".internal"):
        return False, "target host is not publicly routable"

    if _is_non_public_ip(hostname):
        return False, "target host is not publicly routable"

    return True, ""


def _format_fetch_error(exc: Exception) -> str:
    """Return a sanitized fetch error that avoids leaking internal details."""
    import httpx

    if isinstance(exc, httpx.TimeoutException):
        return "[Fetch failed: request timed out]"
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        return f"[Fetch failed: remote server returned HTTP {status_code}]"
    if isinstance(exc, httpx.RequestError):
        return "[Fetch failed: network error while fetching URL]"
    return "[Fetch failed: unexpected error while fetching URL]"


def _normalize_text_key(text: str) -> str:
    """Normalize text for duplicate checks and deterministic ordering."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _result_object_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a search result object (e.g. exa_py.api.Result) to a plain dict."""
    if isinstance(obj, dict):
        return obj
    out: dict[str, Any] = {}
    for attr in ("url", "title", "text", "content", "highlights", "score", "published_date", "author", "summary"):
        val = getattr(obj, attr, None)
        if val is not None:
            out[attr] = val
    return out or {"content": str(obj)}


def _normalize_search_results(raw_output: Any) -> list[dict[str, Any]]:
    """Normalize heterogeneous search output shapes into a list of dicts."""
    if isinstance(raw_output, dict):
        if isinstance(raw_output.get("results"), list):
            return [_result_object_to_dict(item) for item in raw_output["results"]]
        return [raw_output]
    # Handle SearchResponse-style objects with a .results list (e.g. exa_py.api.SearchResponse).
    if not isinstance(raw_output, (list, str)) and hasattr(raw_output, "results"):
        results_attr = raw_output.results
        if isinstance(results_attr, list):
            return [_result_object_to_dict(item) for item in results_attr]
    if isinstance(raw_output, list):
        return [_result_object_to_dict(item) for item in raw_output]
    if isinstance(raw_output, str):
        return [{"content": raw_output}]
    return [{"content": str(raw_output)}]


def _deduplicate_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by URL when present, otherwise by title/content fingerprint."""
    deduped: dict[str, dict[str, Any]] = {}
    for result in results:
        url_key = _normalize_text_key(str(result.get("url") or ""))
        if url_key:
            dedupe_key = f"url:{url_key}"
        else:
            raw_content_key = _normalize_text_key(str(result.get("raw_content") or ""))
            content_key = _normalize_text_key(str(result.get("content") or ""))
            snippet_key = (raw_content_key or content_key)[:240]
            if not snippet_key:
                snippet_key = _normalize_text_key(str(result.get("title") or ""))
            dedupe_key = f"text:{snippet_key}"
        deduped.setdefault(dedupe_key, result)
    return list(deduped.values())


def _sort_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically sort processed search results."""
    return sorted(
        results,
        key=lambda result: (
            1 if not _normalize_text_key(str(result.get("url") or "")) else 0,
            _normalize_text_key(str(result.get("url") or "")),
            _normalize_text_key(str(result.get("title") or "")),
            _normalize_text_key(str(result.get("content") or ""))[:120],
        ),
    )


def _extract_search_snippet(result: dict[str, Any], max_chars: int = 1200) -> str:
    """Extract a bounded text snippet from raw provider search output."""
    highlights = result.get("highlights")
    highlight_chunks: list[str] = []

    if isinstance(highlights, str):
        highlight_chunks = [highlights]
    elif isinstance(highlights, list):
        for item in highlights:
            if isinstance(item, str):
                highlight_chunks.append(item)
            elif isinstance(item, dict):
                for key in ("text", "highlight", "content", "snippet"):
                    value = item.get(key)
                    if value:
                        highlight_chunks.append(str(value))
                        break
            elif item is not None:
                highlight_chunks.append(str(item))

            if len(highlight_chunks) >= 6:
                break

    highlight_text = "\n".join(chunk.strip() for chunk in highlight_chunks if str(chunk).strip()).strip()
    if highlight_text:
        return highlight_text[:max_chars]

    raw_content = str(result.get("raw_content") or "").strip()
    if raw_content:
        return raw_content[:max_chars]

    content = str(result.get("content") or "").strip()
    if content:
        return content[:max_chars]

    text = str(result.get("text") or "").strip()
    if text:
        return text[:max_chars]

    return "[No content available]"


def _format_search_results_for_agent(results: list[dict[str, str]]) -> str:
    """Format processed search results for tool output."""
    if not results:
        return "No relevant search results found."

    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        lines.append(f"[Source {idx}] {result.get('title', 'Untitled Source')}")
        lines.append(f"URL: {result.get('url', 'N/A')}")
        lines.append(result.get("summary", ""))
        lines.append("-" * 40)
    return "\n".join(lines).strip()


def _count_raw_result_items(raw_output: Any) -> int:
    """Best-effort count of raw result objects returned by search provider."""
    if raw_output is None:
        return 0
    if isinstance(raw_output, list):
        return len(raw_output)
    if isinstance(raw_output, dict) and isinstance(raw_output.get("results"), list):
        return len(raw_output["results"])
    # Handle SearchResponse-style objects with a .results list.
    if hasattr(raw_output, "results") and isinstance(raw_output.results, list):
        return len(raw_output.results)
    return 1


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Record strategic reasoning between searches.

    Args:
        reflection: Short reasoning note on findings, gaps, and next search step.
    """
    return f"Reflection recorded: {reflection}"


def _build_fetch_url_tool(
    writer: Callable[[dict[str, Any]], None] | None = None,
):
    """Build a fetch_url tool that extracts main content from web pages."""

    def emit(event: dict[str, Any]) -> None:
        if writer is not None:
            writer(event)

    @tool(parse_docstring=True)
    async def fetch_url(url: str) -> str:
        """Fetch and extract the main content from a web page URL.

        Use this to get full article/page content when search result snippets
        are insufficient. Only fetch URLs that appeared in search results.

        Args:
            url: The URL to fetch content from.
        """
        import httpx

        max_chars = 8000
        is_valid_target, blocked_reason = _validate_fetch_url_target(url)
        if not is_valid_target:
            emit({"event": "fetch_url_blocked", "url": url, "reason": blocked_reason})
            return f"[Fetch blocked: {blocked_reason}]"

        emit({"event": "fetch_url", "url": url})

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepResearchBot/1.0)"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
        except Exception as exc:
            return _format_fetch_error(exc)

        content: str | None = None
        try:
            import trafilatura

            content = trafilatura.extract(html, include_links=True)
        except Exception:
            content = None

        if not content:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                content = soup.get_text(separator="\n", strip=True)
            except Exception:
                content = None

        if not content:
            return "[Fetch failed: could not extract content from page]"

        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[content truncated]"
        return content

    return fetch_url



def _build_search_tool_with_processing(
    base_search_tool: Any,
    writer: Callable[[dict[str, Any]], None] | None = None,
):
    """Wrap provider search with deterministic normalize→dedupe→truncate/format."""

    def emit(event: dict[str, Any]) -> None:
        if writer is not None:
            writer(event)

    async def invoke_search(payload: Any) -> Any:
        if hasattr(base_search_tool, "ainvoke"):
            return await base_search_tool.ainvoke(payload)
        return base_search_tool.invoke(payload)

    @tool("search_web", parse_docstring=True)
    async def search_web(query: str) -> str:
        """Search the web for a query and return deduplicated source summaries.

        Args:
            query: Search query string.
        """
        if base_search_tool is None:
            return "Search is unavailable in this run."

        raw_output = None
        errors: list[str] = []

        is_exa_search = base_search_tool.__class__.__name__ == "ExaSearchResults"
        payload_candidates: list[Any] = []
        if is_exa_search:
            payload_candidates.append(
                {
                    "query": query,
                    "num_results": 5,
                    "highlights": True,
                    "type": "auto",
                }
            )
        payload_candidates.append({"query": query})
        payload_candidates.append(query)

        for payload in payload_candidates:
            try:
                raw_output = await invoke_search(payload)
            except Exception as exc:
                errors.append(str(exc))
                continue

            if isinstance(raw_output, dict) and raw_output.get("error"):
                return f"Search failed for '{query}': {raw_output.get('error')}"

            if raw_output is not None:
                break

        if raw_output is None:
            return f"Search failed for '{query}': {' | '.join(errors)}"

        raw_count = _count_raw_result_items(raw_output)
        normalized = _normalize_search_results(raw_output)
        deduped = _deduplicate_search_results(normalized)
        ordered = _sort_search_results(deduped)
        processed = [
            {
                "title": str(result.get("title", "Untitled Source")),
                "url": str(result.get("url", "N/A")),
                "summary": _extract_search_snippet(result),
            }
            for result in ordered[:MAX_SEARCH_RESULTS_FOR_AGENT]
        ]

        emit(
            {
                "event": "search_preprocess",
                "query": query,
                "raw_result_count": raw_count,
                "normalized_count": len(normalized),
                "deduped_count": len(deduped),
                "returned_count": len(processed),
                "llm_calls_in_preprocess": 0,
            }
        )
        return _format_search_results_for_agent(processed)

    return search_web
