"""Specialist Agent — generates one debate turn for a specialist persona."""

import json

from config import MAX_ARGUMENT_TOKENS
from llm.base import LLMClient
from models.schemas import AgentPersona, DebateState, DebateTurn, Claim
from prompts.specialist_prompt import SPECIALIST_SYSTEM, SPECIALIST_USER, ROUTER_PROMPT
from agents.utils import format_transcript, format_claim_ledger, get_last_mediator_summary


class SpecialistAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def generate_turn(self, agent: AgentPersona, state: DebateState) -> DebateTurn:
        """Generate one complete debate turn for the given specialist."""
        system = SPECIALIST_SYSTEM.format(
            name=agent.name,
            specialty=agent.specialty,
            side=agent.side,
            background=agent.background,
            personality=agent.personality,
            core_argument_angle=agent.core_argument_angle,
        )
        user = SPECIALIST_USER.format(
            topic=state.topic,
            current_turn=state.current_turn,
            total_turns=state.total_turns,
            transcript=format_transcript(state.turns),
            claim_ledger=format_claim_ledger(state.claim_ledger),
            last_mediator_summary=get_last_mediator_summary(state),
            name=agent.name,
            side=agent.side,
        )

        thinking, raw_text = self.client.chat(
            system, user, MAX_ARGUMENT_TOKENS, enable_thinking=True
        )
        return self._parse_turn(thinking, raw_text, agent, state.current_turn)

    def route_next_speaker(
        self,
        last_turn: DebateTurn,
        candidates: list[AgentPersona],
        response_side: str,
    ) -> AgentPersona:
        """Lightweight routing call: pick which specialist responds best to the last argument."""
        if len(candidates) == 1:
            return candidates[0]

        specialists_list = "\n".join(
            f"  - {a.name}: {a.specialty} ({a.personality})" for a in candidates
        )
        prompt = ROUTER_PROMPT.format(
            last_speaker=last_turn.agent_name,
            last_side=last_turn.side,
            last_argument=last_turn.argument[:500],
            response_side=response_side,
            specialists_list=specialists_list,
        )
        try:
            chosen_name = self.client.chat_simple(prompt, max_tokens=100)
            for agent in candidates:
                if agent.name.lower() in chosen_name.lower():
                    return agent
        except Exception:
            pass
        return candidates[0]

    def _parse_turn(
        self,
        thinking: str,
        raw_text: str,
        agent: AgentPersona,
        turn_number: int,
    ) -> DebateTurn:
        data = self._safe_json_parse(raw_text)

        new_claims: list[Claim] = []
        for c in data.get("new_claims", []):
            new_claims.append(Claim(
                claim_id=c.get("claim_id", f"claim_{turn_number:03d}"),
                agent_name=agent.name,
                side=agent.side,
                content=c.get("content", ""),
                turn_number=turn_number,
                rebutted_by=[],
            ))

        return DebateTurn(
            turn_number=turn_number,
            agent_name=agent.name,
            agent_role=agent.specialty,
            side=agent.side,
            thinking=thinking or data.get("thinking", ""),
            argument=data.get("argument", ""),
            new_claims=new_claims,
            rebuttal_targets=data.get("rebuttal_targets", []),
            confidence=float(data.get("confidence", 0.5)),
        )

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
        fixed = self.client.chat_simple(
            f"The following JSON is malformed. Fix it and return ONLY valid JSON, nothing else:\n\n{malformed}",
            max_tokens=4096,
        )
        fixed = fixed.strip()
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            fixed = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
        return json.loads(fixed)
