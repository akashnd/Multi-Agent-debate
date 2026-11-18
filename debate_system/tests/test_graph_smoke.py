"""End-to-end smoke test for the debate graph, with no API calls.

Every chat model is swapped for a fake that returns canned structured output, so
this exercises the parts that are actually easy to get wrong — routing, the
claim-ledger reducer, mediator scheduling, human interrupts and resumption,
transcript persistence — in about a second and for free.

Run:  python -m pytest -q   (or)   python tests/test_graph_smoke.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LANGSMITH_TRACING"] = "false"

import pytest  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from models.schemas import AgentPersona, Claim  # noqa: E402
from tests.fakes import TOPIC, install_fake_models  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _fakes() -> None:
    install_fake_models()


@pytest.fixture(scope="module")
def graph_run():
    """Drive the graph with a human participant; yield its final state and stats."""
    from graph.builder import SERDE, build_debate_graph, initial_state

    graph = build_debate_graph(checkpointer=InMemorySaver(serde=SERDE))
    human = AgentPersona(
        name="You",
        side="con",
        specialty="Human Participant",
        background="A human.",
        personality="Speaks for themselves.",
        core_argument_angle="lived experience",
        is_human=True,
    )
    config = {"configurable": {"thread_id": "smoke-1"}, "recursion_limit": 200}
    payload = initial_state(
        topic=TOPIC, n_rounds=3, k_specialists=2, human_persona=human, auto_approve=False
    )

    approvals = ["regenerate", "approve"]  # exercise the regenerate loop once
    human_turns = 0
    seen_nodes: list[str] = []
    approval_sizes: list[int] = []

    while True:
        pending = None
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                pending = chunk["__interrupt__"][0].value
                break
            seen_nodes.extend(chunk.keys())
        if pending is None:
            break

        if pending["type"] == "persona_approval":
            approval_sizes.append(len(pending["agents"]))
            payload = Command(resume={"decision": approvals.pop(0)})
        elif pending["type"] == "human_turn":
            human_turns += 1
            payload = Command(resume={"argument": f"Human point #{human_turns}."})
        else:
            raise AssertionError(pending)

    values = graph.get_state(config).values
    yield {
        "graph": graph,
        "values": values,
        "seen_nodes": seen_nodes,
        "human_turns": human_turns,
        "approvals_left": approvals,
        "approval_sizes": approval_sizes,
    }

    path = Path(values["transcript_path"])
    if path.exists():
        path.unlink()  # keep the transcripts/ dir clean


def _speaking(run):
    return [t for t in run["values"]["turns"] if t.side != "mediator"]


def test_claim_ledger_reducer_upserts_and_is_pure() -> None:
    from graph.state import merge_claim_ledger

    base = [Claim(claim_id="claim_001", agent_name="A", side="pro", content="x", turn_number=1)]
    patch = [
        Claim(
            claim_id="claim_001",
            agent_name="A",
            side="pro",
            content="x",
            turn_number=1,
            rebutted_by=["claim_002"],
        )
    ]
    merged = merge_claim_ledger(base, patch)
    assert len(merged) == 1 and merged[0].rebutted_by == ["claim_002"], merged
    assert base[0].rebutted_by == [], "reducer must not mutate its inputs"


def test_graph_compiles_with_all_nodes() -> None:
    from graph.builder import SERDE, build_debate_graph

    graph = build_debate_graph(checkpointer=InMemorySaver(serde=SERDE))
    nodes = set(graph.get_graph().nodes)
    for expected in (
        "decompose",
        "approve",
        "supervisor",
        "specialist",
        "human",
        "mediator",
        "verdict",
        "persist",
    ):
        assert expected in nodes, f"missing node {expected}"


def test_persona_approval_and_regenerate_loop(graph_run) -> None:
    assert not graph_run["approvals_left"], "regenerate loop never ran"
    assert graph_run["seen_nodes"].count("decompose") == 2, (
        "regenerate should re-run decompose exactly once"
    )
    # 4 specialists + human + mediator
    assert graph_run["approval_sizes"] == [6, 6]


def test_turn_counts_and_human_participation(graph_run) -> None:
    values = graph_run["values"]
    specs = [a for a in values["agents"] if a.side != "mediator"]
    assert len(specs) == 5  # 4 generated + 1 human
    assert values["total_turns"] == 5 * 3
    assert len(_speaking(graph_run)) == 15
    assert graph_run["human_turns"] == 3, "human should speak once per round"


def test_everyone_speaks_once_per_round(graph_run) -> None:
    speaking = _speaking(graph_run)
    for r in range(3):
        names = {t.agent_name for t in speaking[r * 5 : (r + 1) * 5]}
        assert len(names) == 5, f"round {r} had duplicate speakers: {names}"


def test_sides_alternate(graph_run) -> None:
    sides = [t.side for t in _speaking(graph_run)]
    alternations = sum(1 for a, b in zip(sides, sides[1:], strict=False) if a != b)
    assert alternations >= 9, f"sides barely alternate: {sides}"


def test_turn_numbers_strictly_increase(graph_run) -> None:
    numbers = [t.turn_number for t in graph_run["values"]["turns"]]
    assert numbers == sorted(set(numbers)), numbers


def test_mediator_cadence(graph_run) -> None:
    values = graph_run["values"]
    summaries = values["mediator_summaries"]
    assert 1 <= len(summaries) <= 4, f"unexpected mediator cadence: {len(summaries)}"
    assert len([t for t in values["turns"] if t.side == "mediator"]) == len(summaries)


def test_claim_ledger_and_rebuttals(graph_run) -> None:
    ledger = graph_run["values"]["claim_ledger"]
    assert len(ledger) == len(_speaking(graph_run)), "one claim per speaking turn"
    assert len({c.claim_id for c in ledger}) == len(ledger), "duplicate claim_ids"
    assert any(c.rebutted_by for c in ledger), "no claim was ever marked rebutted"
    assert all(c.claim_id != r for c in ledger for r in c.rebutted_by), "claim rebuts itself"

    # A claim may only be rebutted by the other side.
    side_of = {c.claim_id: c.side for c in ledger}
    for c in ledger:
        for r in c.rebutted_by:
            if r in side_of:
                assert side_of[r] != c.side, f"{r} rebuts same-side {c.claim_id}"


def test_verdict_and_transcript_saved(graph_run) -> None:
    values = graph_run["values"]
    verdict = values["verdict"]
    assert verdict is not None and verdict.verdict, "no verdict produced"
    path = Path(values["transcript_path"])
    assert path.exists(), path
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["topic"] == TOPIC
    assert len(saved["turns"]) == len(values["turns"])
    assert saved["final_verdict"]["verdict"] == verdict.verdict


def test_cli_driver_renders_and_produces_verdict() -> None:
    """Run the CLI driver end to end against the fakes, with output swallowed.

    Catches the class of bug the graph test can't see: a display call that blows
    up on a real turn, or a node-name typo in the renderer.
    """
    from debate.runner import DebateRunner
    from ui.display import DebateDisplay

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        runner = DebateRunner(DebateDisplay(), thread_id="smoke-driver")
        verdict = runner.run(topic=TOPIC, n_rounds=2, k_specialists=2, auto_approve=True)

    assert verdict is not None and verdict.verdict, "driver produced no verdict"
    rendered = sink.getvalue()
    for expected in ("DEBATE BEGINS", "Ada Pro", "FINAL VERDICT", "Transcript saved"):
        assert expected in rendered, f"display never rendered {expected!r}"

    saved = Path(runner._values()["transcript_path"])
    assert saved.exists()
    saved.unlink()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
