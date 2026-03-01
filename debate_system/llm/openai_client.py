"""OpenAI client — GPT-4o and o-series reasoning models."""

import time

try:
    from openai import OpenAI, APIError as OpenAIAPIError
except ImportError:
    raise ImportError("openai package is required for OpenAI provider. Run: pip install openai")

from llm.base import LLMClient

# o-series models use max_completion_tokens and expose reasoning internally (not extractable)
_O_SERIES_PREFIXES = ("o1", "o3", "o4")


def _is_o_series(model: str) -> bool:
    return any(model.startswith(p) for p in _O_SERIES_PREFIXES)


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str, max_retries: int, retry_base_delay: float):
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        enable_thinking: bool = False,
    ) -> tuple[str, str]:
        """
        OpenAI does not expose reasoning tokens externally for o-series models,
        so thinking_text is always returned as "".
        """
        messages = []
        if system:
            # o-series models accept 'developer' role; standard models use 'system'
            role = "developer" if _is_o_series(self.model) else "system"
            messages.append({"role": role, "content": system})
        messages.append({"role": "user", "content": user})

        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict = dict(
                    model=self.model,
                    messages=messages,
                )
                # o-series uses max_completion_tokens; standard uses max_tokens
                if _is_o_series(self.model):
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

                response = self._client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""
                return "", text

            except OpenAIAPIError as e:
                last_error = e
                time.sleep(self.retry_base_delay * (2 ** attempt))

        raise RuntimeError(
            f"OpenAI API call failed after {self.max_retries} retries: {last_error}"
        )
