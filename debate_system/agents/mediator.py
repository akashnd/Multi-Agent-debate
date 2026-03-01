"""Mediator Agent — mid-debate summaries and final verdict."""

import json

from config import MAX_VERDICT_TOKENS, MAX_SUMMARY_TOKENS
from llm.base import LLMClient
from models.schemas import AgentPersona, DebateState, DebateTurn, MediatorSummary, FinalVerdict, ClaimScore
from prompts.mediator_prompt import MEDIATOR_SYSTEM, MEDIATOR_SUMMARY_USER, MEDIATOR_VERDICT_USER
from agents.utils import format_transcript, format_claim_ledger, format_summaries


class MediatorAgent:
    def __init__(self, client: LLMClient, persona: AgentPersona):
        self.client = client
        self.persona = persona

    def generate_summary_as_turn(self, state: DebateState) -> DebateTurn:
        """Generate a mid-debate summary and return it as a DebateTurn for display."""
        system = MEDIATOR_SYSTEM.format(
            name=self.persona.name,
            background=self.persona.background,
            personality=self.persona.personality,
            core_argument_angle=self.persona.core_argument_angle,
        )
        user = MEDIATOR_SUMMARY_USER.format(
            topic=state.topic,
            transcript=format_transcript(state.turns),
            claim_ledger=format_claim_ledger(state.claim_ledger),
        )

        thinking, raw_text = self.client.chat(system, user, MAX_SUMMARY_TOKENS, enable_thinking=True)
        data = self._safe_json_parse(raw_text)

        summary = MediatorSummary(
            turn_number=state.current_turn,
            state_of_debate=data.get("state_of_debate", ""),
            contested_ground=data.get("contested_ground", ""),
            weakest_arguments=data.get("weakest_arguments", ""),
            probing_question=data.get("probing_question", ""),
        )
        state.mediator_summaries.append(summary)

        argument = (
            f"STATE OF DEBATE: {summary.state_of_debate}\n\n"
            f"CONTESTED GROUND: {summary.contested_ground}\n\n"
            f"WEAKEST ARGUMENTS: {summary.weakest_arguments}\n\n"
            f"PROBING QUESTION: {summary.probing_question}"
        )

        return DebateTurn(
            turn_number=state.current_turn,
            agent_name=self.persona.name,
            agent_role=self.persona.specialty,
            side="mediator",
            thinking=thinking or data.get("thinking", ""),
            argument=argument,
            new_claims=[],
            rebuttal_targets=[],
            confidence=1.0,
        )

    def generate_verdict(self, state: DebateState) -> FinalVerdict:
        """Generate the final scored verdict after all debate turns."""
        system = MEDIATOR_SYSTEM.format(
            name=self.persona.name,
            background=self.persona.background,
            personality=self.persona.personality,
            core_argument_angle=self.persona.core_argument_angle,
        )
        user = MEDIATOR_VERDICT_USER.format(
            topic=state.topic,
            transcript=format_transcript(state.turns),
            claim_ledger=format_claim_ledger(state.claim_ledger),
            summaries=format_summaries(state.mediator_summaries),
        )

        _, raw_text = self.client.chat(system, user, MAX_VERDICT_TOKENS, enable_thinking=True)
        data = self._safe_json_parse(raw_text)

        scorecard: list[ClaimScore] = []
        for item in data.get("argument_scorecard", []):
            scorecard.append(ClaimScore(
                claim_id=item.get("claim_id", ""),
                claim_summary=item.get("claim_summary", ""),
                defense_rating=int(item.get("defense_rating", 3)),
                successfully_rebutted=bool(item.get("successfully_rebutted", False)),
                notes=item.get("notes", ""),
            ))

        return FinalVerdict(
            argument_scorecard=scorecard,
            strongest_pro_arguments=data.get("strongest_pro_arguments", []),
            strongest_con_arguments=data.get("strongest_con_arguments", []),
            fatal_pro_weaknesses=data.get("fatal_pro_weaknesses", ""),
            fatal_con_weaknesses=data.get("fatal_con_weaknesses", ""),
            confidence_trajectory=data.get("confidence_trajectory", ""),
            nuanced_conclusion=data.get("nuanced_conclusion", ""),
            verdict=data.get("verdict", ""),
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
