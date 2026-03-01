"""Abstract base class for all LLM provider clients."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Unified interface over Anthropic, OpenAI, and Google Gemini.

    All implementations handle their own retry logic internally.
    Returns (thinking_text, response_text) from chat().
    thinking_text is empty string for providers that don't expose reasoning.
    """

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        enable_thinking: bool = False,
    ) -> tuple[str, str]:
        """
        Send a chat message and return (thinking_text, response_text).

        Args:
            system: System/instruction prompt.
            user: User message.
            max_tokens: Maximum output tokens.
            enable_thinking: Request extended reasoning if the provider supports it.

        Returns:
            (thinking_text, response_text) — thinking is "" when unsupported.
        """

    def chat_simple(self, user: str, max_tokens: int = 1024) -> str:
        """Convenience wrapper for single-turn calls that don't need thinking."""
        _, text = self.chat("", user, max_tokens, enable_thinking=False)
        return text
