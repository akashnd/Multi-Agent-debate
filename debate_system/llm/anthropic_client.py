"""Anthropic Claude client — supports extended thinking."""

import time
import anthropic

from llm.base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, thinking_budget: int, max_retries: int, retry_base_delay: float):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        enable_thinking: bool = False,
    ) -> tuple[str, str]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": user}],
                )
                if system:
                    kwargs["system"] = system
                if enable_thinking:
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget,
                    }

                response = self._client.messages.create(**kwargs)

                thinking_text = ""
                response_text = ""
                for block in response.content:
                    if block.type == "thinking":
                        thinking_text = block.thinking
                    elif block.type == "text":
                        response_text = block.text

                return thinking_text, response_text

            except anthropic.APIError as e:
                last_error = e
                time.sleep(self.retry_base_delay * (2 ** attempt))

        raise RuntimeError(
            f"Anthropic API call failed after {self.max_retries} retries: {last_error}"
        )
