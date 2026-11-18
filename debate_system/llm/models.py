"""LangChain chat-model factory.

One entry point — `get_chat_model()` — returns a configured `BaseChatModel` for
whichever provider `LLM_PROVIDER` names. Everything downstream talks LCEL:

    chain = prompt | get_chat_model(max_tokens=...).with_structured_output(Schema)
    result: Schema = chain.invoke({...})

so provider differences (message format, tool-calling dialect, retries) are
handled by LangChain rather than by hand-rolled client wrappers.
"""

from __future__ import annotations

import os
import sys
from functools import cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    ANTHROPIC_MODEL,
    GEMINI_MODEL,
    MAX_RETRIES,
    MODEL_TEMPERATURE,
    OPENAI_MODEL,
)

# provider -> (langchain provider id, default model, required env var)
_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "anthropic": ("anthropic", ANTHROPIC_MODEL, "ANTHROPIC_API_KEY"),
    "openai": ("openai", OPENAI_MODEL, "OPENAI_API_KEY"),
    "gemini": ("google_genai", GEMINI_MODEL, "GOOGLE_API_KEY"),
}


def active_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower().strip()


@cache
def get_chat_model(
    max_tokens: int = 4096,
    temperature: float = MODEL_TEMPERATURE,
    provider: str | None = None,
) -> BaseChatModel:
    """Build (and memoise) a chat model for the configured provider.

    Cached on its arguments so the handful of distinct configurations used by
    the graph — one per token budget — are each instantiated once.
    """
    name = (provider or active_provider()).lower().strip()

    if name not in _PROVIDERS:
        print(
            f"[ERROR] Unknown LLM_PROVIDER='{name}'. Valid options: {', '.join(_PROVIDERS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    lc_provider, model_name, env_var = _PROVIDERS[name]
    if not os.environ.get(env_var):
        print(
            f"[ERROR] LLM_PROVIDER is set to '{name}' but {env_var} is not set.\n"
            f"Add it to your .env file: {env_var}=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)

    return init_chat_model(
        model=model_name,
        model_provider=lc_provider,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=MAX_RETRIES,
    )
