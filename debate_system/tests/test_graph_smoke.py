"""End-to-end smoke test for the debate graph, with no API calls.

Every chat model is swapped for a fake that returns canned structured output, so
this exercises the parts that are actually easy to get wrong — routing, the
claim-ledger reducer, mediator scheduling, human interrupts and resumption,
transcript persistence — in about a second and for free.

Run:  python tests/test_graph_smoke.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agents.decomposer as decomposer_mod
import agents.mediator as mediator_mod
import agents.specialist as specialist_mod
import agents.supervisor as supervisor_mod
from models.outputs import (
    FinalVerdictOutput,
    MediatorSummaryOutput,
    PersonaSet,
    PersonaSpec,
    ProposedClaim,
    SpecialistTurnOutput,
    SpeakerChoice,
)
from models.schemas import AgentPersona, ClaimScore

TOPIC = "Nuclear power should be the backbone of the grid"


# ──────────────────────────────────────────────────────────────────────────
# Fake model
# ──────────────────────────────────────────────────────────────────────────

def _persona(name: str, side: str, specialty: str) -> PersonaSpec:
    return PersonaSpec(
        name=name,
        side=side,
        specialty=specialty,
        background=f"{name} has spent 20 years on {specialty}.",
        personality="Terse empiricist.",
        core_argument_angle=f"{specialty} lens on the topic",
    )


def _fake_output(schema, prompt_text: str):
    if schema is PersonaSet:
        return PersonaSet(
            thinking="picked two dimensions",
            agents=[
                _persona("Ada Pro", "pro", "Grid Economics"),
                _persona("Ben Con", "con", "Grid Economics"),
                _persona("Cara Pro", "pro", "Safety Engineering"),
                _persona("Dan Con", "con", "Safety Engineering"),
            ],
            mediator=_persona("Mira Neutral", "mediator", "Neutral Synthesis"),
        )

    if schema is SpeakerChoice:
        # Parse the candidate list exactly as the real router prompt renders it.
        names = re.findall(r"^\s+- ([^:]+):", prompt_text, flags=re.MULTILINE)
        return SpeakerChoice(speaker=names[-1] if names else "", reason="fake routing")

    if schema is SpecialistTurnOutput:
        speaker = re.search(r"Speak as ([^,]+), once", prompt_text)
        who = speaker.group(1) if speaker else "someone"
        targets = re.findall(r"(claim_\d+)", prompt_text)
        return SpecialistTurnOutput(
            thinking=f"{who} plans an attack",
            argument=f"{who} argues a point.",
            new_claims=[ProposedClaim(content=f"{who}'s claim")],
            rebuttal_targets=sorted(set(targets)),
            confidence=0.6,
            citations=[],
        )

    if schema is MediatorSummaryOutput:
        return MediatorSummaryOutput(
            thinking="assessing",
            state_of_debate="Both sides have made claims.",
            contested_ground="Cost versus safety.",
            weakest_arguments="Everyone is hand-waving.",
            probing_question="Ada Pro, what is the capital cost per MWh?",
        )

    if schema is FinalVerdictOutput:
        return FinalVerdictOutput(
            thinking="weighing it up",
            argument_scorecard=[
                ClaimScore(
                    claim_id="claim_001",
                    claim_summary="opening claim",
                    defense_rating=3,
                    successfully_rebutted=True,
                    notes="contested",
                )
            ],
            strongest_pro_arguments=["cost curve"],
            strongest_con_arguments=["waste"],
            fatal_pro_weaknesses="no cost data",
            fatal_con_weaknesses="no alternative",
            confidence_trajectory="flat",
            nuanced_conclusion="It depends.",
            verdict="Narrow pro win.",
            citations=[],
        )

    raise AssertionError(f"unexpected schema {schema}")


class _FakeModel:
    def with_structured_output(self, schema, **_kwargs):
        return RunnableLambda(lambda pv: _fake_output(schema, pv.to_string()))


def _install_fakes() -> None:
    fake = lambda *a, **k: _FakeModel()  # noqa: E731
    for module in (decomposer_mod, specialist_mod, mediator_mod, supervisor_mod):
        module.get_chat_model = fake
    # Never touch the network for evidence either.
    import search.tavily as search_mod

    search_mod.get_searcher.cache_clear()
    search_mod.get_searcher = lambda: search_mod.EvidenceSearch.__new__(
        search_mod.EvidenceSearch
    )
    stub = search_mod.get_searcher()
    stub.enabled = False
    stub.tool = None
    for module in (specialist_mod, mediator_mod):
        module.get_searcher = lambda s=stub: s


# ──────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────

def run() -> None:
    _install_fakes()

    from graph.builder import build_debate_graph, initial_state
    from graph.state import merge_claim_ledger
    from models.schemas import Claim

    # ── reducer unit check ────────────────────────────────────────────────
    base = [Claim(claim_id="claim_001", agent_name="A", side="pro", content="x", turn_number=1)]
    patch = [Claim(claim_id="claim_001", agent_name="A", side="pro", content="x", turn_number=1,
                   rebutted_by=["claim_002"])]
    merged = merge_claim_ledger(base, patch)
    assert len(merged) == 1 and merged[0].rebutted_by == ["claim_002"], merged
    assert base[0].rebutted_by == [], "reducer must not mutate its inputs"
    print("[ok] claim-ledger reducer upserts and stays pure")

    # ── graph compiles ────────────────────────────────────────────────────
    from graph.builder import SERDE

    graph = build_debate_graph(checkpointer=InMemorySaver(serde=SERDE))
    nodes = set(graph.get_graph().nodes)
    for expected in ("decompose", "approve", "supervisor", "specialist", "human", "mediator",
                     "verdict", "persist"):
        assert expected in nodes, f"missing node {expected}"
    print("[ok] graph compiles with all", len(nodes), "nodes")

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
            assert len(pending["agents"]) == 6, pending["agents"]  # 4 specialists + human + mediator
            payload = Command(resume={"decision": approvals.pop(0)})
        elif pending["type"] == "human_turn":
            human_turns += 1
            payload = Command(resume={"argument": f"Human point #{human_turns}."})
        else:
            raise AssertionError(pending)

    values = graph.get_state(config).values

    # ── assertions ────────────────────────────────────────────────────────
    assert not approvals, "regenerate loop never ran"
    assert seen_nodes.count("decompose") == 2, "regenerate should re-run decompose exactly once"
    print("[ok] persona approval interrupt + regenerate loop")

    specs = [a for a in values["agents"] if a.side != "mediator"]
    assert len(specs) == 5, specs                       # 4 generated + 1 human
    assert values["total_turns"] == 5 * 3, values["total_turns"]

    speaking = [t for t in values["turns"] if t.side != "mediator"]
    assert len(speaking) == 15, f"expected 15 speaking turns, got {len(speaking)}"
    assert human_turns == 3, f"human should speak once per round, got {human_turns}"
    print(f"[ok] {len(speaking)} speaking turns over 3 rounds, {human_turns} of them human")

    # Everyone speaks exactly once per round.
    for r in range(3):
        names = {t.agent_name for t in speaking[r * 5:(r + 1) * 5]}
        assert len(names) == 5, f"round {r} had duplicate speakers: {names}"
    print("[ok] every specialist speaks exactly once per round")

    # Sides alternate wherever both sides still have eligible speakers.
    sides = [t.side for t in speaking]
    alternations = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
    assert alternations >= 9, f"sides barely alternate: {sides}"
    print("[ok] sides alternate:", "".join("P" if s == "pro" else "C" for s in sides))

    # Turn numbers are strictly increasing across the whole record, mediator included.
    numbers = [t.turn_number for t in values["turns"]]
    assert numbers == sorted(set(numbers)), numbers

    summaries = values["mediator_summaries"]
    assert 1 <= len(summaries) <= 4, f"unexpected mediator cadence: {len(summaries)}"
    assert len([t for t in values["turns"] if t.side == "mediator"]) == len(summaries)
    print(f"[ok] mediator interjected {len(summaries)}x")

    ledger = values["claim_ledger"]
    assert len(ledger) == len(speaking), f"one claim per speaking turn: {len(ledger)}"
    assert len({c.claim_id for c in ledger}) == len(ledger), "duplicate claim_ids"
    rebutted = [c for c in ledger if c.rebutted_by]
    assert rebutted, "no claim was ever marked rebutted"
    assert all(c.claim_id != r for c in ledger for r in c.rebutted_by), "claim rebuts itself"
    print(f"[ok] ledger: {len(ledger)} claims, {len(rebutted)} marked rebutted")

    # A claim may only be rebutted by the other side.
    side_of = {c.claim_id: c.side for c in ledger}
    for c in ledger:
        for r in c.rebutted_by:
            if r in side_of:
                assert side_of[r] != c.side, f"{r} rebuts same-side {c.claim_id}"
    print("[ok] rebuttals only ever cross sides")

    verdict = values["verdict"]
    assert verdict is not None and verdict.verdict, "no verdict produced"
    path = Path(values["transcript_path"])
    assert path.exists(), path
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["topic"] == TOPIC
    assert len(saved["turns"]) == len(values["turns"])
    assert saved["final_verdict"]["verdict"] == verdict.verdict
    print(f"[ok] verdict rendered and transcript saved -> {path.name}")

    path.unlink()  # keep the transcripts/ dir clean

    check_driver()
    print("\nALL CHECKS PASSED")


def check_driver() -> None:
    """Run the CLI driver end to end against the fakes, with output swallowed.

    Catches the class of bug the graph test can't see: a display call that blows
    up on a real turn, or a node-name typo in the renderer.
    """
    import contextlib
    import io

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
    print(f"[ok] CLI driver rendered {len(rendered.splitlines())} lines and produced a verdict")


if __name__ == "__main__":
    run()
