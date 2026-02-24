import pytest


@pytest.fixture(autouse=True)
def _disable_langsmith_tracing_by_default(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
