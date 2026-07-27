"""Graph wiring.

    START -> decompose -> approve ─┬─ regenerate ─> decompose  (loop)
                                   ├─ abort ──────> END
                                   └─ approve ────> supervisor
                                                       │
        ┌──────────────────────────────────────────────┤
        │   supervisor ─┬─ specialist ─┐               │
        │               ├─ human ──────┤               │
        │               ├─ mediator ───┘ (back to supervisor)
        │               └─ verdict ────> persist ─> END
        └──────────────────────────────────────────────┘

The cycle is the point: there is no precomputed turn queue anywhere. Every hop
back into `supervisor` is a fresh decision made against current state, so the
debate's shape is emergent rather than scripted.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from config import CHECKPOINT_DB
from graph.nodes import (
    approve_personas_node,
    decompose_node,
    human_node,
    mediator_node,
    persist_node,
    route_after_approval,
    route_from_supervisor,
    specialist_node,
    supervisor_node,
    verdict_node,
)
from graph.state import DebateGraphState
from models.schemas import (
    AgentPersona,
    Citation,
    Claim,
    ClaimScore,
    DebateState,
    DebateTurn,
    FinalVerdict,
    MediatorSummary,
)

# Our domain models travel in and out of the checkpoint, so the serializer has
# to be told they're safe to reconstruct — otherwise every save/load warns (and
# in a future LangGraph release, refuses).
SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        AgentPersona,
        Citation,
        Claim,
        ClaimScore,
        DebateState,
        DebateTurn,
        FinalVerdict,
        MediatorSummary,
    ]
)


def make_checkpointer() -> BaseCheckpointSaver:
    """InMemorySaver by default; SqliteSaver if CHECKPOINT_DB points at a file.

    The checkpointer is what makes `interrupt()` work at all — the graph's state
    has to be durable across the pause. Point CHECKPOINT_DB at a .sqlite file and
    a debate also survives the process dying, so it can be resumed later on the
    same thread_id.
    """
    if CHECKPOINT_DB and CHECKPOINT_DB.lower() not in ("memory", ":memory:", ""):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            saver = SqliteSaver.from_conn_string(CHECKPOINT_DB).__enter__()
            saver.serde = SERDE
            return saver
        except ImportError:
            print(
                "[warn] CHECKPOINT_DB is set but langgraph-checkpoint-sqlite is not "
                "installed — falling back to in-memory checkpointing."
            )

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver(serde=SERDE)


def build_debate_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the debate graph."""
    builder = StateGraph(DebateGraphState)

    builder.add_node("decompose", decompose_node)
    builder.add_node("approve", approve_personas_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("specialist", specialist_node)
    builder.add_node("human", human_node)
    builder.add_node("mediator", mediator_node)
    builder.add_node("verdict", verdict_node)
    builder.add_node("persist", persist_node)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "approve")

    builder.add_conditional_edges(
        "approve",
        route_after_approval,
        {"approve": "supervisor", "regenerate": "decompose", "abort": END},
    )

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "specialist": "specialist",
            "human": "human",
            "mediator": "mediator",
            "verdict": "verdict",
        },
    )

    # Every speaker hands the floor straight back to the supervisor.
    builder.add_edge("specialist", "supervisor")
    builder.add_edge("human", "supervisor")
    builder.add_edge("mediator", "supervisor")

    builder.add_edge("verdict", "persist")
    builder.add_edge("persist", END)

    return builder.compile(checkpointer=checkpointer or make_checkpointer())


def initial_state(
    topic: str,
    n_rounds: int,
    k_specialists: int,
    human_persona: AgentPersona | None = None,
    auto_approve: bool = False,
) -> DebateGraphState:
    """The state the graph starts from. `total_turns` is filled in by decompose."""
    return {
        "topic": topic,
        "n_rounds": n_rounds,
        "k_specialists": k_specialists,
        "human_persona": human_persona,
        "auto_approve": auto_approve,
        "agents": [],
        "regenerate_count": 0,
        "turns": [],
        "claim_ledger": [],
        "mediator_summaries": [],
        "total_turns": 0,
        "current_turn": 0,
        "specialist_turns_done": 0,
        "round_index": 0,
        "spoken_this_round": [],
        "next_speaker": None,
        "route": "approve",
        "routing_reason": "",
        "verdict": None,
        "transcript_path": None,
    }
