"""Google Gemini client — supports thought extraction on thinking-capable models."""

import time

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPIError
except ImportError:
    raise ImportError(
        "google-generativeai package is required for Gemini provider. "
        "Run: pip install google-generativeai"
    )

from llm.base import LLMClient

# Models that return extractable thought blocks in their response parts
_THINKING_MODELS = {
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-04-17",
}


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str, max_retries: int, retry_base_delay: float):
        genai.configure(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._supports_thinking = model in _THINKING_MODELS

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
                model_kwargs: dict = dict(model_name=self.model)
                if system:
                    model_kwargs["system_instruction"] = system

                model_obj = genai.GenerativeModel(**model_kwargs)

                generation_config = genai.GenerationConfig(max_output_tokens=max_tokens)
                response = model_obj.generate_content(
                    user,
                    generation_config=generation_config,
                )

                # Try to extract thought parts if this is a thinking model
                thinking_text = ""
                text_parts: list[str] = []

                if enable_thinking and self._supports_thinking:
                    try:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, "thought") and part.thought:
                                thinking_text = part.text
                            else:
                                text_parts.append(part.text)
                    except Exception:
                        text_parts = [response.text]
                else:
                    text_parts = [response.text]

                return thinking_text, "\n".join(text_parts)

            except GoogleAPIError as e:
                last_error = e
                time.sleep(self.retry_base_delay * (2 ** attempt))
            except Exception as e:
                # Catch other Gemini errors (e.g. safety blocks)
                last_error = e
                time.sleep(self.retry_base_delay * (2 ** attempt))

        raise RuntimeError(
            f"Gemini API call failed after {self.max_retries} retries: {last_error}"
        )
