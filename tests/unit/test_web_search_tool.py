"""Unit tests for src/archi/pipelines/agents/tools/web_search.py.

All tests mock the underlying langchain-community wrappers so no live network
call is made.  Tests run with importlib isolation so they don't need the full
application stack (no DB, no Flask).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Test helpers — stub out langchain-community providers
# ---------------------------------------------------------------------------


def _install_tavily_stub(payload):
    """Install a fake langchain_community.tools.tavily_search module."""
    fake_module = types.ModuleType("langchain_community.tools.tavily_search")

    class _FakeTavily:
        def __init__(self, max_results=5):
            self.max_results = max_results

        def invoke(self, _):
            return payload

    fake_module.TavilySearchResults = _FakeTavily
    sys.modules["langchain_community.tools.tavily_search"] = fake_module
    return fake_module


def _install_duckduckgo_stub(payload):
    """Install a fake langchain_community.utilities module."""
    fake_module = types.ModuleType("langchain_community.utilities")

    class _FakeDDG:
        def __init__(self, max_results=5):
            self.max_results = max_results

        def results(self, _query, max_results=None):
            return payload

    fake_module.DuckDuckGoSearchAPIWrapper = _FakeDDG
    sys.modules["langchain_community.utilities"] = fake_module
    return fake_module


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Each test starts without TAVILY_API_KEY unless it sets it explicitly."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    yield


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def test_auto_resolves_to_duckduckgo_without_api_key():
    from src.archi.pipelines.agents.tools.web_search import _resolve_provider

    assert _resolve_provider("auto") == "duckduckgo"


def test_auto_resolves_to_tavily_with_api_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "sk-test")
    from src.archi.pipelines.agents.tools.web_search import _resolve_provider

    assert _resolve_provider("auto") == "tavily"


def test_explicit_provider_overrides_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "sk-test")
    from src.archi.pipelines.agents.tools.web_search import _resolve_provider

    assert _resolve_provider("duckduckgo") == "duckduckgo"


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def test_format_results_empty_payload():
    from src.archi.pipelines.agents.tools.web_search import _format_results

    assert _format_results([], provider="tavily") == "No web results found."


def test_format_results_renders_tavily_payload():
    from src.archi.pipelines.agents.tools.web_search import _format_results

    text = _format_results(
        [
            {"title": "Hello", "url": "https://example.com/a", "content": "First match."},
            {"title": "World", "url": "https://example.com/b", "content": "Second match."},
        ],
        provider="tavily",
    )
    assert "provider=tavily" in text
    assert "1. Hello" in text
    assert "https://example.com/a" in text
    assert "First match." in text
    assert "2. World" in text


def test_format_results_renders_duckduckgo_payload():
    """DuckDuckGo's results() returns 'link' + 'snippet' rather than 'url' + 'content'."""
    from src.archi.pipelines.agents.tools.web_search import _format_results

    text = _format_results(
        [{"title": "DDG Hit", "link": "https://example.com/c", "snippet": "From DDG."}],
        provider="duckduckgo",
    )
    assert "DDG Hit" in text
    assert "https://example.com/c" in text
    assert "From DDG." in text


def test_format_results_truncates_long_snippets():
    from src.archi.pipelines.agents.tools.web_search import _format_results

    long = "x" * 1000
    text = _format_results(
        [{"title": "T", "url": "https://example.com", "content": long}],
        provider="tavily",
    )
    # 400-char cap with trailing ellipsis (3 chars).
    assert "..." in text
    assert len(text) < 1200


# ---------------------------------------------------------------------------
# Tool factory end-to-end (with stubs)
# ---------------------------------------------------------------------------


def test_create_web_search_tool_tavily_path(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "sk-test")
    _install_tavily_stub(
        [{"title": "Hit", "url": "https://example.com", "content": "snippet text"}]
    )

    # Re-import module to pick up the stub cleanly.
    import importlib

    import src.archi.pipelines.agents.tools.web_search as ws

    importlib.reload(ws)

    tool = ws.create_web_search_tool(provider="tavily", max_results=3)
    assert tool.name == "web_search"

    result = tool.invoke({"query": "hello"})
    assert "Hit" in result
    assert "https://example.com" in result


def test_create_web_search_tool_duckduckgo_path():
    _install_duckduckgo_stub(
        [{"title": "DDG Hit", "link": "https://example.com/x", "snippet": "snippet"}]
    )
    import importlib
    import src.archi.pipelines.agents.tools.web_search as ws

    importlib.reload(ws)

    tool = ws.create_web_search_tool(provider="duckduckgo", max_results=2)
    assert tool.name == "web_search"
    result = tool.invoke({"query": "hello"})
    assert "DDG Hit" in result


def test_tool_rejects_empty_query():
    _install_duckduckgo_stub([])
    import importlib
    import src.archi.pipelines.agents.tools.web_search as ws

    importlib.reload(ws)

    tool = ws.create_web_search_tool(provider="duckduckgo")
    assert "ERROR" in tool.invoke({"query": ""})


def test_tool_returns_error_string_on_backend_exception():
    fake_module = types.ModuleType("langchain_community.utilities")

    class _ExplodingDDG:
        def __init__(self, max_results=5):
            pass

        def results(self, _query, max_results=None):
            raise RuntimeError("boom")

    fake_module.DuckDuckGoSearchAPIWrapper = _ExplodingDDG
    sys.modules["langchain_community.utilities"] = fake_module

    import importlib
    import src.archi.pipelines.agents.tools.web_search as ws

    importlib.reload(ws)
    tool = ws.create_web_search_tool(provider="duckduckgo")
    out = tool.invoke({"query": "anything"})
    assert "boom" in out
    assert "duckduckgo" in out


def test_tavily_without_api_key_raises():
    import importlib
    import src.archi.pipelines.agents.tools.web_search as ws

    importlib.reload(ws)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        ws.create_web_search_tool(provider="tavily")
