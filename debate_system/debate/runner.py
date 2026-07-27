"""CLI driver for the debate graph.

The graph owns the orchestration; this owns the terminal. It streams node
updates out of `graph.stream(..., stream_mode="updates")`, renders each one, and
answers any `interrupt()` the graph raises by asking the person at the keyboard
and resuming with `Command(resume=...)`.

Swapping this file for a web handler is the whole extent of "make it a service" —
nothing in graph/ or agents/ would change.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterator

from langgraph.types import Command

from config import GRAPH_RECURSION_LIMIT
from graph.builder import build_debate_graph, initial_state
from graph.nodes import save_transcript
from graph.state import to_debate_state
from models.schemas import AgentPersona, FinalVerdict
from ui.display import DebateDisplay, console


class DebateRunner:
    def __init__(self, display: DebateDisplay, thread_id: str | None = None):
        self.display = display
        self.graph = build_debate_graph()
        self.thread_id = thread_id or f"debate-{uuid.uuid4().hex[:8]}"
        self.config: dict[str, Any] = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        self._started = False

    # ── public API ────────────────────────────────────────────────────────

    def run(
        self,
        topic: str,
        n_rounds: int,
        k_specialists: int,
        human_persona: AgentPersona | None = None,
        auto_approve: bool = False,
    ) -> FinalVerdict | None:
        payload: Any = initial_state(
            topic=topic,
            n_rounds=n_rounds,
            k_specialists=k_specialists,
            human_persona=human_persona,
            auto_approve=auto_approve,
        )

        try:
            while True:
                pending = self._drain(self.graph.stream(payload, self.config, stream_mode="updates"))
                if pending is None:
                    break

                resume_value = self._answer_interrupt(pending)
                if resume_value is None:
                    console.print("[dim]Exiting.[/dim]")
                    return None
                payload = Command(resume=resume_value)
        except KeyboardInterrupt:
            self.display.show_interrupt_notice()
            self.display.show_transcript_saved(save_transcript(self._values()))
            return None

        return self._values().get("verdict")

    # ── streaming ─────────────────────────────────────────────────────────

    def _drain(self, stream: Iterator[dict]) -> Any:
        """Render node updates until the graph finishes or interrupts.

        Returns the pending interrupt payload, or None if the graph ran to END.
        """
        for chunk in stream:
            if "__interrupt__" in chunk:
                interrupts = chunk["__interrupt__"]
                return interrupts[0].value if interrupts else None
            for node, update in chunk.items():
                self._render(node, update or {})
        return None

    def _render(self, node: str, update: dict) -> None:
        if node == "decompose":
            agents = update.get("agents", [])
            self.display.show_agents_generated(agents)
            for agent in agents:
                self.display.show_agent_details(agent)

        elif node == "supervisor":
            route = update.get("route")
            speaker = update.get("next_speaker")
            if route == "verdict":
                state = self._debate_state()
                if not self._started:
                    self._started = True
                self.display.show_thinking_spinner(
                    next((a.name for a in state.agents if a.side == "mediator"), "Mediator")
                )
            elif speaker:
                if not self._started:
                    self.display.show_debate_start(self._debate_state())
                    self._started = True
                if route != "human":
                    self.display.show_thinking_spinner(speaker)

        elif node in ("specialist", "human"):
            turns = update.get("turns", [])
            if turns:
                self.display.show_specialist_turn(turns[-1], self._debate_state())

        elif node == "mediator":
            turns = update.get("turns", [])
            if turns:
                self.display.show_mediator_turn(turns[-1])

        elif node == "verdict":
            verdict = update.get("verdict")
            if verdict is not None:
                self.display.show_final_verdict(verdict, self._debate_state())

        elif node == "persist":
            path = update.get("transcript_path")
            if path:
                self.display.show_transcript_saved(path)

    # ── interrupts ────────────────────────────────────────────────────────

    def _answer_interrupt(self, payload: Any) -> Any:
        """Turn a graph interrupt into a console prompt. None means 'abort'."""
        kind = payload.get("type") if isinstance(payload, dict) else None

        if kind == "persona_approval":
            answer = input("Proceed with these agents? [Y/n/r (regenerate)]: ").strip().lower()
            if answer in ("", "y", "yes"):
                return {"decision": "approve"}
            if answer in ("r", "regen", "regenerate"):
                console.print("[dim]Regenerating agents...[/dim]")
                return {"decision": "regenerate"}
            return None

        if kind == "human_turn":
            state = self._debate_state()
            agent = next(
                (a for a in state.agents if a.name == payload.get("agent_name")), None
            )
            if agent is not None:
                state.current_turn = payload.get("turn_number", state.current_turn)
                self.display.show_human_turn_context(agent, state)
            return {"argument": self._read_argument()}

        # Unknown interrupt: resume with nothing rather than deadlocking.
        return {}

    @staticmethod
    def _read_argument() -> str:
        """Read a multi-line argument from stdin. Submit with an empty line."""
        print("Type your argument. Press Enter on an empty line to submit.\n")
        while True:
            lines: list[str] = []
            while True:
                try:
                    line = input("> ")
                except EOFError:
                    break
                if line == "":
                    if lines:
                        break
                    continue  # ignore leading blank lines
                lines.append(line)

            argument = "\n".join(lines).strip()
            if argument:
                return argument
            print("An argument can't be empty. Try again.\n")

    # ── state access ──────────────────────────────────────────────────────

    def _values(self) -> dict:
        snapshot = self.graph.get_state(self.config)
        return snapshot.values or {}

    def _debate_state(self):
        return to_debate_state(self._values())
