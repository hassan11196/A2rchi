"""Web search tool for archi agents.

Provides a langchain-compatible tool that runs a web search via one of two
providers:

* **tavily** — `TavilySearchResults` from langchain-community.  Requires the
  `TAVILY_API_KEY` secret.  Returns clean structured results with snippets.
* **duckduckgo** — `DuckDuckGoSearchAPIWrapper` from langchain-community.  No
  API key required; suitable as a no-credential default and as a fallback when
  Tavily is unavailable.

The factory `create_web_search_tool` returns a `BaseTool` with a stable
`name="web_search"`, so any agent registry can expose it under the same
selectable name regardless of which provider is active.  The tool is
read-only, so it should be classified ``safe`` by any tool-approval gate.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Literal, Optional

from langchain.tools import BaseTool, tool as tool_decorator

from src.utils.env import read_secret
from src.utils.logging import get_logger

logger = get_logger(__name__)

_TOOL_NAME = "web_search"
_TOOL_DESCRIPTION = (
    "Run a live web search and return the top results as a numbered list with "
    "title, URL, and a short snippet. Use this to look up information that is "
    "not in archi's indexed knowledge base, e.g. recent events, current "
    "documentation, package versions, or external references. Input must be a "
    "concise plain-text query (about 3-12 keywords). The tool returns one "
    "block of plain text; cite returned URLs verbatim when quoting them."
)

ProviderName = Literal["auto", "tavily", "duckduckgo"]


def _resolve_provider(provider: ProviderName) -> str:
    """Pick a concrete provider, defaulting to tavily when a key exists."""
    if provider != "auto":
        return provider
    if read_secret("TAVILY_API_KEY"):
        return "tavily"
    return "duckduckgo"


def _build_tavily_runner(max_results: int) -> Callable[[str], str]:
    """Return a callable(query) -> str backed by Tavily."""
    api_key = read_secret("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "web_search provider='tavily' selected but TAVILY_API_KEY is not set"
        )
    # Pass through env for langchain-community's own discovery.
    os.environ.setdefault("TAVILY_API_KEY", api_key)
    from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore

    backend = TavilySearchResults(max_results=max_results)

    def _run(query: str) -> str:
        results: List[dict] = backend.invoke({"query": query}) or []
        return _format_results(results, provider="tavily")

    return _run


def _build_duckduckgo_runner(max_results: int) -> Callable[[str], str]:
    """Return a callable(query) -> str backed by DuckDuckGo."""
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper  # type: ignore

    backend = DuckDuckGoSearchAPIWrapper(max_results=max_results)

    def _run(query: str) -> str:
        results: List[dict] = backend.results(query, max_results=max_results) or []
        return _format_results(results, provider="duckduckgo")

    return _run


def _format_results(results: List[dict], *, provider: str) -> str:
    """Render a uniform plain-text block from heterogeneous provider payloads."""
    if not results:
        return "No web results found."

    lines: List[str] = [f"web_search results (provider={provider}):"]
    for idx, item in enumerate(results, start=1):
        # Provider payloads differ slightly: Tavily uses content/url/title,
        # DuckDuckGo's results() uses snippet/link/title.
        title = item.get("title") or "(no title)"
        url = item.get("url") or item.get("link") or ""
        snippet = item.get("content") or item.get("snippet") or ""
        snippet = snippet.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:397].rstrip() + "..."
        lines.append(f"{idx}. {title}")
        if url:
            lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def create_web_search_tool(
    *,
    provider: ProviderName = "auto",
    max_results: int = 5,
) -> BaseTool:
    """Build a langchain `web_search` tool backed by Tavily or DuckDuckGo.

    Args:
        provider: ``"tavily"``, ``"duckduckgo"``, or ``"auto"`` (Tavily if
            ``TAVILY_API_KEY`` is set, else DuckDuckGo).
        max_results: Maximum number of result rows to render.  Provider
            wrappers may return fewer.

    Returns:
        A `BaseTool` with `name="web_search"` and a stable description.
    """
    resolved = _resolve_provider(provider)
    if resolved == "tavily":
        runner = _build_tavily_runner(max_results)
    elif resolved == "duckduckgo":
        runner = _build_duckduckgo_runner(max_results)
    else:  # pragma: no cover — _resolve_provider rejects unknown values
        raise ValueError(f"Unknown web_search provider: {resolved!r}")

    logger.info("Built web_search tool (provider=%s, max_results=%d)", resolved, max_results)

    @tool_decorator(_TOOL_NAME, description=_TOOL_DESCRIPTION)
    def web_search(query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "ERROR: 'query' is required."
        try:
            return runner(query)
        except Exception as exc:
            logger.warning("web_search (%s) failed: %s", resolved, exc)
            return f"web_search error ({resolved}): {exc}"

    return web_search
