"""Tavily web search client for grounding debate arguments with real sources."""

import os
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


class TavilySearch:
    """Wraps the Tavily API to fetch evidence for debate agents."""

    def __init__(self):
        self._api_key = os.environ.get("TAVILY_API_KEY", "")
        search_enabled_env = os.environ.get("SEARCH_ENABLED", "true").lower()
        self.enabled = bool(self._api_key) and search_enabled_env != "false"
        self._client = None

        if self.enabled:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
            except ImportError:
                self.enabled = False

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        """Search for evidence. Returns [] if disabled or on any error."""
        if not self.enabled or self._client is None:
            return []
        try:
            response = self._client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
            )
            results = []
            for r in response.get("results", []):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", "")[:500],  # keep snippets concise
                    score=float(r.get("score", 0.0)),
                ))
            return results
        except Exception:
            return []

    def format_for_prompt(self, results: list[SearchResult]) -> str:
        """Format results for injection into an LLM prompt."""
        if not results:
            return "(No search results — no TAVILY_API_KEY set or search disabled.)"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.title}\n"
                f"    URL: {r.url}\n"
                f"    {r.content}"
            )
        return "\n\n".join(lines)
