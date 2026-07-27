"""Human participant.

The human node doesn't read stdin — it calls LangGraph's `interrupt()`, which
suspends the graph and hands control back to whatever is driving it (the CLI
today, a web frontend tomorrow). This module just turns the resumed text into a
proper `DebateTurn` plus its ledger delta.
"""

from __future__ import annotations

from agents.utils import next_claim_id
from models.schemas import AgentPersona, Claim, DebateState, DebateTurn


def build_human_turn(
    agent: AgentPersona,
    state: DebateState,
    turn_number: int,
    argument: str,
) -> tuple[DebateTurn, list[Claim]]:
    """Wrap a human's typed argument as a turn. Returns (turn, ledger_delta).

    Deliberately minimal: the system assigns the claim_id and registers the
    argument as one claim. No thinking step, no rebuttal-target picking, no
    citations — those are LLM-specialist niceties this skips on purpose.
    """
    argument = (argument or "").strip()

    claim = Claim(
        claim_id=next_claim_id(state.claim_ledger),
        agent_name=agent.name,
        side=agent.side,
        content=argument,
        turn_number=turn_number,
        rebutted_by=[],
    )

    turn = DebateTurn(
        turn_number=turn_number,
        agent_name=agent.name,
        agent_role=agent.specialty,
        side=agent.side,
        thinking="",
        argument=argument,
        new_claims=[claim],
        rebuttal_targets=[],
        confidence=0.5,
        citations=[],
    )
    return turn, [claim]
