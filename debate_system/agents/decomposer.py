"""Topic Decomposition Agent — generates specialist personas for a debate topic."""

import json

from llm.base import LLMClient
from models.schemas import AgentPersona
from prompts.decomposer_prompt import DECOMPOSER_SYSTEM, DECOMPOSER_USER


class TopicDecomposer:
    def __init__(self, client: LLMClient):
        self.client = client

    def decompose(self, topic: str, k: int) -> list[AgentPersona]:
        """Generate K specialist pairs + mediator for the given topic."""
        prompt = DECOMPOSER_USER.format(topic=topic, k=k)
        _, raw_json = self.client.chat(DECOMPOSER_SYSTEM, prompt, max_tokens=4096)
        return self._parse_personas(raw_json)

    def _parse_personas(self, raw: str) -> list[AgentPersona]:
        """Parse JSON response into AgentPersona objects, with repair fallback."""
        data = self._safe_json_parse(raw)

        personas: list[AgentPersona] = []
        for agent_data in data.get("agents", []):
            personas.append(AgentPersona(**agent_data))

        mediator_data = data.get("mediator")
        if mediator_data:
            personas.append(AgentPersona(**mediator_data))

        return personas

    def _safe_json_parse(self, raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return self._repair_json(raw)

    def _repair_json(self, malformed: str) -> dict:
        """Ask the LLM to fix malformed JSON."""
        fixed = self.client.chat_simple(
            f"The following JSON is malformed. Fix it and return ONLY valid JSON, nothing else:\n\n{malformed}",
            max_tokens=4096,
        )
        fixed = fixed.strip()
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            fixed = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
        return json.loads(fixed)
