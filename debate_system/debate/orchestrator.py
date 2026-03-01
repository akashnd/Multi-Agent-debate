"""Debate Orchestrator — the brain that runs the full debate loop."""

import json
import signal
import datetime
from pathlib import Path

from llm.base import LLMClient
from models.schemas import AgentPersona, DebateState, DebateTurn, FinalVerdict
from agents.specialist import SpecialistAgent
from agents.mediator import MediatorAgent
from debate.turn_selector import TurnSelector
from ui.display import DebateDisplay
from search.tavily import TavilySearch


class DebateOrchestrator:
    def __init__(
        self,
        client: LLMClient,
        topic: str,
        n_rounds: int,
        agents: list[AgentPersona],
        display: DebateDisplay,
    ):
        self.client = client
        self.display = display

        # Separate mediator from specialists
        mediator_persona = next(a for a in agents if a.side == "mediator")
        specialist_personas = [a for a in agents if a.side != "mediator"]

        # Total turns = specialists per side * 2 * n_rounds
        total_turns = len(specialist_personas) * n_rounds

        self.state = DebateState(
            topic=topic,
            total_turns=total_turns,
            current_turn=0,
            agents=agents,
            turns=[],
            claim_ledger=[],
            mediator_summaries=[],
        )

        # Shared searcher so all agents reuse the same Tavily client
        searcher = TavilySearch()
        self.specialist_agent = SpecialistAgent(client, searcher)
        self.mediator_agent = MediatorAgent(client, mediator_persona, searcher)
        self.turn_selector = TurnSelector(agents, total_turns)
        self._interrupted = False
        self._setup_interrupt_handler()

    def _setup_interrupt_handler(self):
        """Handle Ctrl+C gracefully: save partial transcript and exit."""
        def handler(sig, frame):
            self._interrupted = True
            self.display.show_interrupt_notice()
            self._save_transcript()
            raise SystemExit(0)
        signal.signal(signal.SIGINT, handler)

    def run(self) -> FinalVerdict | None:
        """Run the full debate and return the final verdict."""
        self.display.show_debate_start(self.state)

        turn_number = 0
        last_turn: DebateTurn | None = None

        # Build the flat sequence of agent turns across all rounds
        turn_queue: list[AgentPersona] = []
        n_rounds = self.state.total_turns // max(
            1, len([a for a in self.state.agents if a.side != "mediator"])
        )

        for _ in range(n_rounds):
            round_queue = self.turn_selector.build_round_queue(last_turn)
            # Reactive routing: if there was a last turn, reorder con/pro accordingly
            if last_turn:
                response_side = "pro" if last_turn.side == "con" else "con"
                candidates = [a for a in round_queue if a.side == response_side]
                if candidates:
                    chosen = self.specialist_agent.route_next_speaker(
                        last_turn, candidates, response_side
                    )
                    round_queue = self.turn_selector.reorder_for_reactive_response(
                        round_queue, chosen
                    )
            turn_queue.extend(round_queue)

        for agent in turn_queue:
            if self._interrupted:
                break

            turn_number += 1
            self.state.current_turn = turn_number

            # Check if mediator summary is due before this speaker
            if self.turn_selector.is_mediator_turn(turn_number - 1) and turn_number > 1:
                self.display.show_thinking_spinner(self.mediator_agent.persona.name)
                mediator_turn = self.mediator_agent.generate_summary_as_turn(self.state)
                mediator_turn.turn_number = turn_number
                self.state.turns.append(mediator_turn)
                self.display.show_mediator_turn(mediator_turn)
                turn_number += 1
                self.state.current_turn = turn_number

            # Generate specialist turn
            self.display.show_thinking_spinner(agent.name)
            turn_result = self.specialist_agent.generate_turn(agent, self.state)
            turn_result.turn_number = turn_number
            self.state.turns.append(turn_result)
            self._update_claim_ledger(turn_result)
            last_turn = turn_result
            self.display.show_specialist_turn(turn_result, self.state)

        if self._interrupted:
            return None

        # Final mediator verdict
        self.display.show_thinking_spinner(
            next(a.name for a in self.state.agents if a.side == "mediator")
        )
        verdict = self.mediator_agent.generate_verdict(self.state)
        self.display.show_final_verdict(verdict, self.state)
        self._save_transcript(verdict)
        return verdict

    def _update_claim_ledger(self, turn: DebateTurn):
        """Add new claims to ledger and mark rebutted claims."""
        # Add new claims
        for claim in turn.new_claims:
            self.state.claim_ledger.append(claim)

        # Mark rebutted claims
        for rebutted_id in turn.rebuttal_targets:
            for claim in self.state.claim_ledger:
                if claim.claim_id == rebutted_id:
                    claim.rebutted_by.append(
                        turn.new_claims[0].claim_id if turn.new_claims else turn.agent_name
                    )

    def _save_transcript(self, verdict: FinalVerdict | None = None):
        """Save the debate transcript to a JSON file."""
        transcript_dir = Path(__file__).parent.parent / "transcripts"
        transcript_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = transcript_dir / f"debate_transcript_{timestamp}.json"

        data = self.state.model_dump()
        if verdict:
            data["final_verdict"] = verdict.model_dump()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.display.show_transcript_saved(str(filename))
