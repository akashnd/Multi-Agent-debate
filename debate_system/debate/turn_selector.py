"""Dynamic turn order selection for the debate."""

import random
from models.schemas import AgentPersona, DebateTurn
from config import MEDIATOR_SUMMARY_INTERVAL


class TurnSelector:
    """Determines the order in which agents speak each round."""

    def __init__(self, agents: list[AgentPersona], total_turns: int):
        self.pro_agents = [a for a in agents if a.side == "pro"]
        self.con_agents = [a for a in agents if a.side == "con"]
        self.mediator = next((a for a in agents if a.side == "mediator"), None)
        self.total_turns = total_turns
        self._turn_queue: list[AgentPersona] = []
        self._mediator_due_at: set[int] = self._compute_mediator_turns()

    def _compute_mediator_turns(self) -> set[int]:
        """Pre-compute which turn numbers get a mediator summary."""
        interval = max(1, self.total_turns // 3)
        turns = set()
        t = interval
        while t < self.total_turns:
            turns.add(t)
            t += interval
        return turns

    def is_mediator_turn(self, turn_number: int) -> bool:
        return turn_number in self._mediator_due_at

    def build_round_queue(self, last_turn: DebateTurn | None = None) -> list[AgentPersona]:
        """
        Build one round of speaker order.
        Alternates pro/con to maximize clash.
        Shuffles specialist order slightly for unpredictability.
        If last_turn is provided, puts the most reactive specialist first.
        """
        pro_order = self.pro_agents.copy()
        con_order = self.con_agents.copy()

        # Slight shuffle to avoid predictability (keep first speaker if reactive routing used)
        if len(pro_order) > 1:
            first_pro = pro_order[0]
            rest = pro_order[1:]
            random.shuffle(rest)
            pro_order = [first_pro] + rest

        if len(con_order) > 1:
            first_con = con_order[0]
            rest = con_order[1:]
            random.shuffle(rest)
            con_order = [first_con] + rest

        # Interleave pro/con
        queue: list[AgentPersona] = []
        for p, c in zip(pro_order, con_order):
            queue.append(p)
            queue.append(c)

        # Handle unequal sides (edge case)
        if len(pro_order) > len(con_order):
            queue.extend(pro_order[len(con_order):])
        elif len(con_order) > len(pro_order):
            queue.extend(con_order[len(pro_order):])

        return queue

    def reorder_for_reactive_response(
        self,
        queue: list[AgentPersona],
        chosen_agent: AgentPersona,
    ) -> list[AgentPersona]:
        """Move the chosen reactive agent to the front of the queue."""
        if chosen_agent in queue:
            queue.remove(chosen_agent)
            queue.insert(0, chosen_agent)
        return queue
