"""Evidence search, backed by the LangChain Tavily tool.

`langchain_tavily.TavilySearch` is a real `BaseTool`, so the same object can be
handed to a tool-calling agent later; here the graph nodes call it directly and
format the hits into prompt context. Search is best-effort: no key, no package
or a failed request all degrade to "no results" rather than killing a turn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


class EvidenceSearch:
    """Thin wrapper over the Tavily LangChain tool."""

    def __init__(self, max_results: int = 3):
        self._api_key = os.environ.get("TAVILY_API_KEY", "")
        search_enabled_env = os.environ.get("SEARCH_ENABLED", "true").lower()
        self.enabled = bool(self._api_key) and search_enabled_env != "false"
        self.tool = None

        if self.enabled:
            try:
                from langchain_tavily import TavilySearch as TavilySearchTool

                self.tool = TavilySearchTool(
                    max_results=max_results,
                    search_depth="basic",
                    include_answer=False,
                    tavily_api_key=self._api_key,
                )
            except Exception:
                self.enabled = False

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        """Search for evidence. Returns [] if disabled or on any error."""
        if not self.enabled or self.tool is None:
            return []
        try:
            response = self.tool.invoke({"query": query})
            raw = response.get("results", []) if isinstance(response, dict) else []
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=(r.get("content") or "")[:500],  # keep snippets concise
                    score=float(r.get("score", 0.0) or 0.0),
                )
                for r in raw[:max_results]
            ]
        except Exception:
            return []

    def format_for_prompt(self, results: list[SearchResult]) -> str:
        """Format results for injection into a prompt."""
        if not results:
            return "(No search results — no TAVILY_API_KEY set or search disabled.)"
        return "\n\n".join(
            f"[{i}] {r.title}\n    URL: {r.url}\n    {r.content}"
            for i, r in enumerate(results, 1)
        )

    def context_for(self, query: str, max_results: int | None = None) -> str:
        """search() + format_for_prompt() — what every node actually wants."""
        from config import TAVILY_MAX_RESULTS

        n = TAVILY_MAX_RESULTS if max_results is None else max_results
        return self.format_for_prompt(self.search(query, max_results=n))


# Name kept for backwards compatibility with the pre-LangGraph code.
TavilySearch = EvidenceSearch


@lru_cache(maxsize=1)
def get_searcher() -> EvidenceSearch:
    """Process-wide shared searcher, so every node reuses one client."""
    from config import TAVILY_MAX_RESULTS

    return EvidenceSearch(max_results=TAVILY_MAX_RESULTS)
