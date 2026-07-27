"""Graph nodes — every step of the debate, as a pure state transition.

Each function takes the graph state and returns only the keys it changes.
Nothing here knows about the console: rendering happens in the driver, which
watches the stream of node updates. That separation is what lets the same graph
run behind a CLI, a web app or a scheduled job.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from langgraph.types import interrupt

from agents.decomposer import generate_personas
from agents.human import build_human_turn
from agents.mediator import build_final_verdict, build_mediator_summary
from agents.specialist import build_specialist_turn
from agents.supervisor import choose_speaker
from config import MEDIATOR_SUMMARY_DIVISOR
from graph.state import (
    DebateGraphState,
    agent_by_name,
    last_specialist_turn,
    mediator_of,
    specialists,
    to_debate_state,
)

TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "transcripts"


# ──────────────────────────────────────────────────────────────────────────
# Cast
# ──────────────────────────────────────────────────────────────────────────

def decompose_node(state: DebateGraphState) -> dict:
    """Invent the specialists, then size the debate around them."""
    personas = generate_personas(state["topic"], state["k_specialists"])

    human = state.get("human_persona")
    if human is not None:
        # Insert before the mediator so the mediator stays last.
        personas.insert(len(personas) - 1, human)

    n_specialists = len([p for p in personas if p.side != "mediator"])
    return {
        "agents": personas,
        "total_turns": n_specialists * state["n_rounds"],
    }


def approve_personas_node(state: DebateGraphState) -> dict:
    """Pause for a human to approve, regenerate or abort the cast.

    This is a separate node from `decompose_node` on purpose. A node that
    interrupts is re-executed from the top when the graph resumes, so keeping
    the (expensive, non-idempotent) LLM call in its own node means approving the
    cast never silently regenerates it.
    """
    if state.get("auto_approve"):
        return {"route": "approve"}

    answer = interrupt(
        {
            "type": "persona_approval",
            "agents": [a.model_dump() for a in state["agents"]],
        }
    )

    decision = answer.get("decision") if isinstance(answer, dict) else answer
    decision = str(decision or "approve").strip().lower()

    if decision in ("r", "regen", "regenerate"):
        return {"route": "regenerate", "regenerate_count": state.get("regenerate_count", 0) + 1}
    if decision in ("n", "no", "abort", "quit", "exit"):
        return {"route": "abort"}
    return {"route": "approve"}


# ──────────────────────────────────────────────────────────────────────────
# Supervisor — the only node that decides anything
# ──────────────────────────────────────────────────────────────────────────

def supervisor_node(state: DebateGraphState) -> dict:
    """Decide what happens next: a specialist, the human, the mediator, or the verdict.

    Deterministic rules (side alternation, one turn each per round, when the
    mediator interjects) live here in code. The single judgement call — which of
    several eligible same-side specialists is best placed to answer what was
    just said — is delegated to the router chain.
    """
    specs = specialists(state)
    total_turns = state["total_turns"]
    round_index = state.get("round_index", 0)
    spoken = list(state.get("spoken_this_round", []))
    specialist_turns_done = state.get("specialist_turns_done", 0)

    # Round rollover: everyone has spoken, so open a new round.
    remaining = [a for a in specs if a.name not in spoken]
    if not remaining:
        round_index += 1
        spoken = []
        remaining = list(specs)

    # Debate over?
    if round_index >= state["n_rounds"] or specialist_turns_done >= total_turns:
        return {
            "route": "verdict",
            "round_index": round_index,
            "spoken_this_round": spoken,
            "next_speaker": None,
        }

    # Mediator interjection: roughly MEDIATOR_SUMMARY_DIVISOR times per debate.
    every = max(1, total_turns // MEDIATOR_SUMMARY_DIVISOR)
    summaries_owed = specialist_turns_done // every
    if specialist_turns_done > 0 and len(state.get("mediator_summaries", [])) < summaries_owed:
        return {
            "route": "mediator",
            "round_index": round_index,
            "spoken_this_round": spoken,
            "next_speaker": mediator_of(state).name,
            "routing_reason": f"mediator checkpoint after {specialist_turns_done} turns",
        }

    # Whose turn: alternate sides, then let the router break the tie.
    last = last_specialist_turn(state)
    target_side = "pro" if last is None else ("con" if last.side == "pro" else "pro")
    candidates = [a for a in remaining if a.side == target_side] or remaining

    agent, reason = choose_speaker(candidates, last, target_side)

    return {
        "route": "human" if agent.is_human else "specialist",
        "next_speaker": agent.name,
        "routing_reason": reason,
        "round_index": round_index,
        "spoken_this_round": spoken + [agent.name],
    }


def route_after_approval(state: DebateGraphState) -> str:
    return state.get("route", "approve")


def route_from_supervisor(state: DebateGraphState) -> str:
    return state.get("route", "verdict")


# ──────────────────────────────────────────────────────────────────────────
# Speaking nodes
# ──────────────────────────────────────────────────────────────────────────

def specialist_node(state: DebateGraphState) -> dict:
    agent = agent_by_name(state, state.get("next_speaker"))
    if agent is None:
        return {"route": "verdict"}

    turn_number = state.get("current_turn", 0) + 1
    turn, ledger_delta = build_specialist_turn(agent, to_debate_state(state), turn_number)

    return {
        "turns": [turn],
        "claim_ledger": ledger_delta,
        "current_turn": turn_number,
        "specialist_turns_done": state.get("specialist_turns_done", 0) + 1,
    }


def human_node(state: DebateGraphState) -> dict:
    """Suspend the graph and wait for a real person to type their turn."""
    agent = agent_by_name(state, state.get("next_speaker"))
    if agent is None:
        return {"route": "verdict"}

    turn_number = state.get("current_turn", 0) + 1

    answer = interrupt(
        {
            "type": "human_turn",
            "agent_name": agent.name,
            "side": agent.side,
            "turn_number": turn_number,
            "total_turns": state["total_turns"],
        }
    )
    argument = answer.get("argument") if isinstance(answer, dict) else answer

    turn, ledger_delta = build_human_turn(
        agent, to_debate_state(state), turn_number, str(argument or "")
    )

    return {
        "turns": [turn],
        "claim_ledger": ledger_delta,
        "current_turn": turn_number,
        "specialist_turns_done": state.get("specialist_turns_done", 0) + 1,
    }


def mediator_node(state: DebateGraphState) -> dict:
    turn_number = state.get("current_turn", 0) + 1
    turn, summary = build_mediator_summary(
        mediator_of(state), to_debate_state(state), turn_number
    )
    return {
        "turns": [turn],
        "mediator_summaries": [summary],
        "current_turn": turn_number,
    }


def verdict_node(state: DebateGraphState) -> dict:
    verdict = build_final_verdict(mediator_of(state), to_debate_state(state))
    return {"verdict": verdict}


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────

def persist_node(state: DebateGraphState) -> dict:
    return {"transcript_path": save_transcript(state)}


def save_transcript(state: DebateGraphState) -> str:
    """Write the debate to transcripts/ as JSON. Safe to call on partial state."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = TRANSCRIPT_DIR / f"debate_transcript_{timestamp}.json"

    data = to_debate_state(state).model_dump()
    verdict = state.get("verdict")
    if verdict is not None:
        data["final_verdict"] = verdict.model_dump()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(path)
