"""The debate's graph state — the single channel set every node reads and writes.

LangGraph merges each node's returned partial dict into this state. Keys with an
`Annotated[..., reducer]` are *accumulated* (nodes return only their delta);
plain keys are last-write-wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from models.schemas import (
    AgentPersona,
    Claim,
    DebateState,
    DebateTurn,
    FinalVerdict,
    MediatorSummary,
)

Route = Literal[
    "specialist",
    "human",
    "mediator",
    "verdict",
    "approve",
    "regenerate",
    "abort",
]


def merge_claim_ledger(left: list[Claim] | None, right: list[Claim] | None) -> list[Claim]:
    """Reducer for the claim ledger — an upsert keyed on `claim_id`.

    A speaking node returns two kinds of claims and this reducer folds both in:
      1. genuinely new claims  -> appended in order
      2. *rebuttal patches*    -> a copy of an existing claim carrying only the
                                  new entries for `rebutted_by`, which get
                                  unioned into the claim already on the record

    Doing it here rather than in the node means a node never has to read-modify-
    write the whole ledger, which is what makes concurrent/parallel debate
    branches safe to add later.
    """
    merged: dict[str, Claim] = {}
    order: list[str] = []

    for claim in list(left or []) + list(right or []):
        existing = merged.get(claim.claim_id)
        if existing is None:
            merged[claim.claim_id] = claim.model_copy(deep=True)
            order.append(claim.claim_id)
            continue
        for rebutter in claim.rebutted_by:
            if rebutter not in existing.rebutted_by:
                existing.rebutted_by.append(rebutter)

    return [merged[cid] for cid in order]


class DebateGraphState(TypedDict, total=False):
    """Everything the debate graph knows.

    Accumulating channels: turns, claim_ledger, mediator_summaries.
    """

    # ── configuration, set once at START ──────────────────────────────────
    topic: str
    n_rounds: int
    k_specialists: int
    human_persona: AgentPersona | None
    auto_approve: bool

    # ── cast ──────────────────────────────────────────────────────────────
    agents: list[AgentPersona]
    regenerate_count: int

    # ── accumulated record ────────────────────────────────────────────────
    turns: Annotated[list[DebateTurn], operator.add]
    claim_ledger: Annotated[list[Claim], merge_claim_ledger]
    mediator_summaries: Annotated[list[MediatorSummary], operator.add]

    # ── supervisor bookkeeping ────────────────────────────────────────────
    total_turns: int
    current_turn: int
    specialist_turns_done: int
    round_index: int
    spoken_this_round: list[str]
    next_speaker: str | None
    route: Route
    routing_reason: str

    # ── output ────────────────────────────────────────────────────────────
    verdict: FinalVerdict | None
    transcript_path: str | None


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def specialists(state: DebateGraphState) -> list[AgentPersona]:
    return [a for a in state.get("agents", []) if a.side != "mediator"]


def mediator_of(state: DebateGraphState) -> AgentPersona:
    return next(a for a in state["agents"] if a.side == "mediator")


def agent_by_name(state: DebateGraphState, name: str | None) -> AgentPersona | None:
    if not name:
        return None
    return next((a for a in state.get("agents", []) if a.name == name), None)


def last_specialist_turn(state: DebateGraphState) -> DebateTurn | None:
    for turn in reversed(state.get("turns", [])):
        if turn.side != "mediator":
            return turn
    return None


def to_debate_state(state: DebateGraphState) -> DebateState:
    """Project the graph state onto the plain `DebateState` model.

    Prompt formatters, the Rich display and the saved transcript all predate the
    graph and speak `DebateState`; this keeps them untouched.
    """
    return DebateState(
        topic=state.get("topic", ""),
        total_turns=state.get("total_turns", 0),
        current_turn=state.get("current_turn", 0),
        agents=state.get("agents", []),
        turns=state.get("turns", []),
        claim_ledger=state.get("claim_ledger", []),
        mediator_summaries=state.get("mediator_summaries", []),
    )
