"""Each turn records which sources the agent was actually shown."""

from __future__ import annotations

import json
import os

os.environ["LANGSMITH_TRACING"] = "false"

import pytest  # noqa: E402

import agents.mediator as mediator_mod  # noqa: E402
import agents.specialist as specialist_mod  # noqa: E402
from models.schemas import AgentPersona, DebateState, DebateTurn, FinalVerdict  # noqa: E402
from search.tavily import EvidenceSearch, SearchResult  # noqa: E402
from tests.fakes import TOPIC, install_fake_models  # noqa: E402


class FakeSearcher(EvidenceSearch):
    def __init__(self) -> None:
        self.enabled = True
        self.tool = None
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 3) -> list[SearchResult]:
        self.queries.append(query)
        return [
            SearchResult(title="Doc A", url="https://a.example/x", content="alpha content"),
            SearchResult(title="Doc B", url="https://b.example/y", content="beta content"),
        ]


@pytest.fixture
def searcher(monkeypatch: pytest.MonkeyPatch) -> FakeSearcher:
    install_fake_models()
    fake = FakeSearcher()
    monkeypatch.setattr(specialist_mod, "get_searcher", lambda: fake)
    monkeypatch.setattr(mediator_mod, "get_searcher", lambda: fake)
    return fake


def _persona(side: str) -> AgentPersona:
    return AgentPersona(
        name="Ada",
        side=side,
        specialty="Economics",
        background="bg",
        personality="p",
        core_argument_angle="angle",
    )


def _state() -> DebateState:
    return DebateState(topic=TOPIC, total_turns=4)


def test_retrieve_returns_results_and_prompt_text(searcher: FakeSearcher) -> None:
    results, text = searcher.retrieve("q")
    assert [r.url for r in results] == ["https://a.example/x", "https://b.example/y"]
    assert "Doc A" in text and "https://b.example/y" in text
    assert searcher.context_for("q") == text  # backward-compatible wrapper


def test_specialist_turn_records_retrieved_sources(searcher: FakeSearcher) -> None:
    turn, _ = specialist_mod.build_specialist_turn(_persona("pro"), _state(), 1)
    assert turn.search_query == f"{TOPIC} Economics"
    assert [(c.title, c.url, c.snippet) for c in turn.retrieved_sources] == [
        ("Doc A", "https://a.example/x", "alpha content"),
        ("Doc B", "https://b.example/y", "beta content"),
    ]


def test_mediator_summary_and_verdict_record_retrieved_sources(searcher: FakeSearcher) -> None:
    mediator = _persona("mediator")
    turn, _ = mediator_mod.build_mediator_summary(mediator, _state(), 2)
    assert len(turn.retrieved_sources) == 2
    assert turn.search_query.endswith("research evidence")

    verdict = mediator_mod.build_final_verdict(mediator, _state())
    assert len(verdict.retrieved_sources) == 2
    assert verdict.search_query.endswith("expert analysis conclusion")


def test_no_search_results_gives_empty_sources() -> None:
    install_fake_models()  # disabled searcher
    turn, _ = specialist_mod.build_specialist_turn(_persona("pro"), _state(), 1)
    assert turn.retrieved_sources == []


def test_old_transcripts_still_load() -> None:
    old_turn = {
        "turn_number": 1,
        "agent_name": "A",
        "agent_role": "r",
        "side": "pro",
        "thinking": "",
        "argument": "x",
    }
    turn = DebateTurn(**json.loads(json.dumps(old_turn)))
    assert turn.retrieved_sources == [] and turn.search_query == ""
    assert FinalVerdict.model_fields["retrieved_sources"].default_factory is list
