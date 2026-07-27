"""Topic decomposition — invents the cast of specialists for a topic.

An LCEL chain: prompt | model.with_structured_output(PersonaSet). The provider
guarantees the shape, so this module only has to sanity-check the *semantics*
(sides balanced, names unique, exactly one mediator).
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from config import MAX_PERSONA_TOKENS
from llm.models import get_chat_model
from models.outputs import PersonaSet, PersonaSpec
from models.schemas import AgentPersona
from prompts.decomposer_prompt import DECOMPOSER_SYSTEM, DECOMPOSER_USER

DECOMPOSER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", DECOMPOSER_SYSTEM), ("human", DECOMPOSER_USER)]
)


def decomposer_chain() -> Runnable:
    """prompt | structured model — topic in, PersonaSet out."""
    model = get_chat_model(max_tokens=MAX_PERSONA_TOKENS)
    return DECOMPOSER_PROMPT | model.with_structured_output(PersonaSet)


def _to_persona(spec: PersonaSpec, forced_side: str | None = None) -> AgentPersona:
    side = (forced_side or spec.side or "").strip().lower()
    if side not in ("pro", "con", "mediator"):
        side = "pro"
    return AgentPersona(
        name=spec.name.strip(),
        side=side,
        specialty=spec.specialty,
        background=spec.background,
        personality=spec.personality,
        core_argument_angle=spec.core_argument_angle,
        is_human=False,
    )


def generate_personas(topic: str, k: int) -> list[AgentPersona]:
    """Return 2*k specialists plus one mediator, in that order."""
    result: PersonaSet = decomposer_chain().invoke({"topic": topic, "k": k})

    personas = [_to_persona(spec) for spec in result.agents]

    # Guard against a model that hands back a lopsided cast.
    if not any(p.side == "pro" for p in personas) or not any(p.side == "con" for p in personas):
        for i, p in enumerate(personas):
            p.side = "pro" if i % 2 == 0 else "con"

    # De-duplicate names, since claim attribution and routing key on them.
    seen: set[str] = set()
    for p in personas:
        base = p.name
        n = 2
        while p.name in seen:
            p.name = f"{base} ({n})"
            n += 1
        seen.add(p.name)

    personas.append(_to_persona(result.mediator, forced_side="mediator"))
    return personas


class TopicDecomposer:
    """Kept for backwards compatibility with the pre-graph entry point."""

    def decompose(self, topic: str, k: int) -> list[AgentPersona]:
        return generate_personas(topic, k)
