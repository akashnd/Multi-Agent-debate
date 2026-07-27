"""Specialist turn generation.

`build_specialist_turn` returns two things: the `DebateTurn` to append to the
record, and the *delta* for the claim ledger. The delta contains both brand-new
claims and "rebuttal patches" — stub copies of existing claims carrying only the
new `rebutted_by` entry — which the ledger reducer folds in. The node never has
to read-modify-write the whole ledger.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from agents.utils import (
    format_claim_ledger,
    format_transcript,
    get_last_mediator_summary,
    next_claim_id,
)
from config import MAX_ARGUMENT_TOKENS
from llm.models import get_chat_model
from models.outputs import SpecialistTurnOutput
from models.schemas import AgentPersona, Citation, Claim, DebateState, DebateTurn
from prompts.specialist_prompt import SPECIALIST_SYSTEM, SPECIALIST_USER
from search.tavily import get_searcher

SPECIALIST_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SPECIALIST_SYSTEM), ("human", SPECIALIST_USER)]
)


def specialist_chain() -> Runnable:
    model = get_chat_model(max_tokens=MAX_ARGUMENT_TOKENS)
    return SPECIALIST_PROMPT | model.with_structured_output(SpecialistTurnOutput)


def build_specialist_turn(
    agent: AgentPersona,
    state: DebateState,
    turn_number: int,
) -> tuple[DebateTurn, list[Claim]]:
    """Generate one specialist turn. Returns (turn, claim_ledger_delta)."""
    search_context = get_searcher().context_for(f"{state.topic} {agent.specialty}")

    out: SpecialistTurnOutput = specialist_chain().invoke(
        {
            # persona (system message)
            "name": agent.name,
            "specialty": agent.specialty,
            "side": agent.side,
            "background": agent.background,
            "personality": agent.personality,
            "core_argument_angle": agent.core_argument_angle,
            # debate context (human message)
            "topic": state.topic,
            "current_turn": turn_number,
            "total_turns": state.total_turns,
            "transcript": format_transcript(state.turns),
            "claim_ledger": format_claim_ledger(state.claim_ledger),
            "last_mediator_summary": get_last_mediator_summary(state),
            "search_results": search_context,
        }
    )

    # Claim ids are assigned here, not by the model — models are unreliable at
    # continuing a sequence and collisions would silently corrupt the ledger.
    ledger_delta: list[Claim] = []
    known = list(state.claim_ledger)
    new_claims: list[Claim] = []
    for proposed in out.new_claims:
        claim = Claim(
            claim_id=next_claim_id(known + new_claims),
            agent_name=agent.name,
            side=agent.side,
            content=proposed.content,
            turn_number=turn_number,
            rebutted_by=[],
        )
        new_claims.append(claim)
    ledger_delta.extend(new_claims)

    # Rebuttal patches: only for claim_ids that actually exist and aren't ours.
    attribution = new_claims[0].claim_id if new_claims else agent.name
    existing_by_id = {c.claim_id: c for c in state.claim_ledger}
    valid_targets: list[str] = []
    for target_id in out.rebuttal_targets:
        target = existing_by_id.get(target_id)
        if target is None or target.side == agent.side:
            continue
        valid_targets.append(target_id)
        ledger_delta.append(target.model_copy(deep=True, update={"rebutted_by": [attribution]}))

    turn = DebateTurn(
        turn_number=turn_number,
        agent_name=agent.name,
        agent_role=agent.specialty,
        side=agent.side,
        thinking=out.thinking,
        argument=out.argument,
        new_claims=new_claims,
        rebuttal_targets=valid_targets,
        confidence=max(0.0, min(1.0, float(out.confidence))),
        citations=[Citation(**c.model_dump()) for c in out.citations],
    )
    return turn, ledger_delta
