"""The supervisor's LLM routing call.

Deciding *who speaks next* is the one genuinely agentic choice in the debate
loop, so it gets its own small structured-output chain rather than being baked
into a precomputed queue. The deterministic parts of turn order — side
alternation and "everyone speaks once per round" — stay in code, where they
belong; the model only breaks ties between eligible same-side specialists.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from config import MAX_ROUTER_TOKENS
from llm.models import get_chat_model
from models.outputs import SpeakerChoice
from models.schemas import AgentPersona, DebateTurn
from prompts.specialist_prompt import ROUTER_PROMPT

ROUTER_CHAT_PROMPT = ChatPromptTemplate.from_messages([("human", ROUTER_PROMPT)])


def router_chain() -> Runnable:
    model = get_chat_model(max_tokens=MAX_ROUTER_TOKENS, temperature=0.0)
    return ROUTER_CHAT_PROMPT | model.with_structured_output(SpeakerChoice)


def choose_speaker(
    candidates: list[AgentPersona],
    last_turn: DebateTurn | None,
    response_side: str,
) -> tuple[AgentPersona, str]:
    """Pick the sharpest responder among `candidates`. Returns (agent, reason)."""
    if not candidates:
        raise ValueError("choose_speaker called with no candidates")
    if len(candidates) == 1:
        return candidates[0], "only eligible speaker on this side"
    if last_turn is None:
        return candidates[0], "opening statement"

    specialists_list = "\n".join(
        f"  - {a.name}: {a.specialty} ({a.personality})" for a in candidates
    )
    try:
        choice: SpeakerChoice = router_chain().invoke(
            {
                "last_speaker": last_turn.agent_name,
                "last_side": last_turn.side,
                "last_argument": last_turn.argument[:800],
                "response_side": response_side,
                "specialists_list": specialists_list,
            }
        )
        picked = choice.speaker.strip().lower()
        for agent in candidates:
            if agent.name.lower() in picked or picked in agent.name.lower():
                return agent, choice.reason
    except Exception:
        pass

    return candidates[0], "routing fell back to queue order"
