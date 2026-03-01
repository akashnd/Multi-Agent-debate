"""Factory — reads env vars and creates the configured LLM client."""

import os
import sys

from llm.base import LLMClient


def create_client() -> LLMClient:
    """
    Instantiate and return the LLM client specified by LLM_PROVIDER env var.

    Supported providers (set LLM_PROVIDER in .env):
        anthropic  (default) — requires ANTHROPIC_API_KEY
        openai               — requires OPENAI_API_KEY
        gemini               — requires GOOGLE_API_KEY

    Optional model overrides:
        ANTHROPIC_MODEL   (default: claude-sonnet-4-6)
        OPENAI_MODEL      (default: gpt-4o)
        GEMINI_MODEL      (default: gemini-1.5-pro)
    """
    from config import (
        ANTHROPIC_MODEL, OPENAI_MODEL, GEMINI_MODEL,
        THINKING_BUDGET, MAX_RETRIES, RETRY_BASE_DELAY,
    )

    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower().strip()

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            _missing_key("ANTHROPIC_API_KEY", "anthropic")
        from llm.anthropic_client import AnthropicClient
        return AnthropicClient(
            api_key=api_key,
            model=ANTHROPIC_MODEL,
            thinking_budget=THINKING_BUDGET,
            max_retries=MAX_RETRIES,
            retry_base_delay=RETRY_BASE_DELAY,
        )

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            _missing_key("OPENAI_API_KEY", "openai")
        from llm.openai_client import OpenAIClient
        return OpenAIClient(
            api_key=api_key,
            model=OPENAI_MODEL,
            max_retries=MAX_RETRIES,
            retry_base_delay=RETRY_BASE_DELAY,
        )

    if provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            _missing_key("GOOGLE_API_KEY", "gemini")
        from llm.gemini_client import GeminiClient
        return GeminiClient(
            api_key=api_key,
            model=GEMINI_MODEL,
            max_retries=MAX_RETRIES,
            retry_base_delay=RETRY_BASE_DELAY,
        )

    print(
        f"[ERROR] Unknown LLM_PROVIDER='{provider}'. "
        "Valid options: anthropic, openai, gemini"
    )
    sys.exit(1)


def _missing_key(env_var: str, provider: str):
    print(
        f"[ERROR] LLM_PROVIDER is set to '{provider}' but {env_var} is not set.\n"
        f"Add it to your .env file: {env_var}=your_key_here"
    )
    sys.exit(1)
