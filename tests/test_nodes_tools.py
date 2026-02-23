import asyncio
from unittest.mock import AsyncMock, patch

from deepresearch import nodes


class _SearchToolRaisesThenString:
    async def ainvoke(self, args, config=None):
        if isinstance(args, dict):
            raise RuntimeError("dict-shape unavailable")
        return "single string provider output"


class _SearchToolReturnsError:
    async def ainvoke(self, args, config=None):
        return {"error": "provider unavailable"}


def test_think_tool_returns_reflection_marker():
    output = nodes.think_tool.invoke({"reflection": "need one more source"})
    assert output == "Reflection recorded: need one more source"


def test_search_preprocessing_handles_malformed_provider_output_types():
    metrics: list[dict] = []
    search_tool = nodes._build_search_tool_with_processing(
        base_search_tool=_SearchToolRaisesThenString(),
        writer=metrics.append,
    )

    output = asyncio.run(search_tool.ainvoke({"query": "test malformed"}))
    assert "[Source 1]" in output
    assert "single string provider output" in output
    assert metrics[-1]["event"] == "search_preprocess"
    assert metrics[-1]["raw_result_count"] == 1
    assert metrics[-1]["deduped_count"] == 1
    assert metrics[-1]["llm_calls_in_preprocess"] == 0


def test_search_preprocessing_surfaces_provider_error_dict():
    search_tool = nodes._build_search_tool_with_processing(
        base_search_tool=_SearchToolReturnsError(),
        writer=lambda event: None,
    )
    output = asyncio.run(search_tool.ainvoke({"query": "test error"}))
    assert output == "Search failed for 'test error': provider unavailable"


def test_search_preprocessing_is_deterministic_and_llm_free():
    class FakeSearchTool:
        async def ainvoke(self, args, config=None):
            if isinstance(args, dict):
                return {
                    "results": [
                        {"title": "B", "url": "https://b.example", "raw_content": "bbbb"},
                        {"title": "A", "url": "https://a.example", "raw_content": "aaaa"},
                        {"title": "A-dup", "url": "https://a.example", "raw_content": "aaaa-dup"},
                        {"title": "No URL 1", "content": "shared text"},
                        {"title": "No URL 2", "content": "shared text"},
                    ]
                }
            raise AssertionError("Unexpected invocation shape")

    metrics: list[dict] = []
    search_tool = nodes._build_search_tool_with_processing(
        base_search_tool=FakeSearchTool(),
        writer=metrics.append,
    )

    first_output = asyncio.run(search_tool.ainvoke({"query": "test"}))
    second_output = asyncio.run(search_tool.ainvoke({"query": "test"}))
    assert first_output == second_output

    last_metric = metrics[-1]
    assert last_metric["event"] == "search_preprocess"
    assert last_metric["raw_result_count"] == 5
    assert last_metric["normalized_count"] == 5
    assert last_metric["deduped_count"] == 3
    assert last_metric["returned_count"] == 3
    assert last_metric["llm_calls_in_preprocess"] == 0


def test_fetch_url_extracts_content_with_trafilatura():
    events: list[dict] = []
    fetch_tool = nodes._build_fetch_url_tool(events.append)

    html = "<html><body><article><p>Main article content here.</p></article></body></html>"
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = html
    mock_response.raise_for_status = lambda: None

    import types

    fake_trafilatura = types.ModuleType("trafilatura")
    fake_trafilatura.extract = lambda html, **kwargs: "Main article content here."

    with (
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.dict("sys.modules", {"trafilatura": fake_trafilatura}),
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(fetch_tool.ainvoke({"url": "https://example.com/article"}))

    assert len(result) > 0
    assert "[Fetch failed" not in result
    assert "Main article content here." in result
    assert any(e.get("event") == "fetch_url" for e in events)


def test_fetch_url_blocks_non_public_targets_before_request():
    events: list[dict] = []
    fetch_tool = nodes._build_fetch_url_tool(events.append)

    with patch("httpx.AsyncClient") as mock_client_cls:
        result = asyncio.run(fetch_tool.ainvoke({"url": "http://127.0.0.1:8080/health"}))

    assert result == "[Fetch blocked: target host is not publicly routable]"
    mock_client_cls.assert_not_called()
    assert any(e.get("event") == "fetch_url_blocked" for e in events)


def test_fetch_url_blocks_non_http_schemes():
    events: list[dict] = []
    fetch_tool = nodes._build_fetch_url_tool(events.append)

    result = asyncio.run(fetch_tool.ainvoke({"url": "file:///etc/passwd"}))

    assert result == "[Fetch blocked: URL scheme must be http or https]"
    assert any(e.get("event") == "fetch_url_blocked" for e in events)


def test_fetch_url_returns_sanitized_error_on_timeout():
    events: list[dict] = []
    fetch_tool = nodes._build_fetch_url_tool(events.append)

    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ReadTimeout("timed out for internal-host"))
        mock_client_cls.return_value = mock_client

        result = asyncio.run(fetch_tool.ainvoke({"url": "https://example.com/slow"}))

    assert result == "[Fetch failed: request timed out]"
    assert "internal-host" not in result
    assert any(e.get("event") == "fetch_url" for e in events)


def test_fetch_url_returns_sanitized_error_on_connect_failure():
    events: list[dict] = []
    fetch_tool = nodes._build_fetch_url_tool(events.append)

    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError(
                "dial tcp 10.0.0.8:443: connect: operation timed out",
                request=httpx.Request("GET", "https://example.com"),
            )
        )
        mock_client_cls.return_value = mock_client

        result = asyncio.run(fetch_tool.ainvoke({"url": "https://example.com/down"}))

    assert result == "[Fetch failed: network error while fetching URL]"
    assert "10.0.0.8" not in result
    assert any(e.get("event") == "fetch_url" for e in events)


def test_fetch_url_truncates_long_content():
    events: list[dict] = []
    fetch_tool = nodes._build_fetch_url_tool(events.append)

    long_text = "A" * 20000
    html = f"<html><body><p>{long_text}</p></body></html>"
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = html
    mock_response.raise_for_status = lambda: None

    import types

    fake_trafilatura = types.ModuleType("trafilatura")
    fake_trafilatura.extract = lambda html, **kwargs: None  # force bs4 fallback

    fake_bs4 = types.ModuleType("bs4")

    class FakeSoup:
        def __init__(self, html, parser):
            self._text = long_text

        def __call__(self, tags):
            return []

        def get_text(self, separator="\n", strip=True):
            return self._text

    fake_bs4.BeautifulSoup = FakeSoup

    with (
        patch("httpx.AsyncClient") as mock_client_cls,
        patch.dict("sys.modules", {"trafilatura": fake_trafilatura, "bs4": fake_bs4}),
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(fetch_tool.ainvoke({"url": "https://example.com/long"}))

    assert result.endswith("...[content truncated]")
    assert len(result) < 8100
