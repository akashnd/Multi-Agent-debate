"""Shared formatting utilities for agents (avoids circular imports)."""

import re

from models.schemas import Claim, DebateTurn, MediatorSummary


def format_transcript(turns: list[DebateTurn]) -> str:
    if not turns:
        return "(No turns yet — you are speaking first.)"
    lines = []
    for t in turns:
        lines.append(
            f"[Turn {t.turn_number}] {t.agent_name} ({t.agent_role}, {t.side.upper()}): "
            f"{t.argument}"
        )
    return "\n\n".join(lines)


def format_claim_ledger(claims: list[Claim]) -> str:
    if not claims:
        return "(No claims registered yet.)"
    lines = []
    for c in claims:
        status = "[REBUTTED]" if c.rebutted_by else "[STANDING]"
        lines.append(f"  {c.claim_id} {status} — {c.agent_name} ({c.side}): {c.content}")
    return "\n".join(lines)


def get_last_mediator_summary(state) -> str:
    """state is DebateState — typed loosely to avoid circular model imports."""
    if not state.mediator_summaries:
        return "(No mediator summary yet.)"
    s = state.mediator_summaries[-1]
    return (
        f"State of Debate: {s.state_of_debate}\n"
        f"Contested Ground: {s.contested_ground}\n"
        f"Probing Question: {s.probing_question}"
    )


def next_claim_id(claims: list[Claim]) -> str:
    """Compute the next sequential claim_id (e.g. 'claim_004' after 'claim_003')."""
    max_n = 0
    for c in claims:
        match = re.search(r"(\d+)\s*$", c.claim_id)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return f"claim_{max_n + 1:03d}"


def format_summaries(summaries: list[MediatorSummary]) -> str:
    if not summaries:
        return "(No prior summaries.)"
    parts = []
    for s in summaries:
        parts.append(
            f"[After turn {s.turn_number}]\n"
            f"State: {s.state_of_debate}\n"
            f"Contested: {s.contested_ground}\n"
            f"Weaknesses: {s.weakest_arguments}\n"
            f"Question: {s.probing_question}"
        )
    return "\n\n".join(parts)
