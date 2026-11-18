"""Fake chat models and a disabled searcher, so the debate graph runs with no network.

Importable by tests and (later) by the eval harness's ``--dry-run``:

    from tests.fakes import install_fake_models
"""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableLambda

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
    SpeakerChoice,
    SpecialistTurnOutput,
)
from models.schemas import ClaimScore

TOPIC = "Nuclear power should be the backbone of the grid"


def _persona(name: str, side: str, specialty: str) -> PersonaSpec:
    return PersonaSpec(
        name=name,
        side=side,
        specialty=specialty,
        background=f"{name} has spent 20 years on {specialty}.",
        personality="Terse empiricist.",
        core_argument_angle=f"{specialty} lens on the topic",
    )


def fake_output(schema, prompt_text: str):
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


class FakeModel:
    def with_structured_output(self, schema, **_kwargs):
        return RunnableLambda(lambda pv: fake_output(schema, pv.to_string()))


def install_fake_models() -> None:
    """Swap every chat model for a fake and disable web search."""

    def fake(*_args, **_kwargs):
        return FakeModel()

    for module in (decomposer_mod, specialist_mod, mediator_mod, supervisor_mod):
        module.get_chat_model = fake

    # Never touch the network for evidence either.
    import search.tavily as search_mod

    search_mod.get_searcher.cache_clear()
    search_mod.get_searcher = lambda: search_mod.EvidenceSearch.__new__(search_mod.EvidenceSearch)
    stub = search_mod.get_searcher()
    stub.enabled = False
    stub.tool = None
    for module in (specialist_mod, mediator_mod):
        module.get_searcher = lambda s=stub: s
