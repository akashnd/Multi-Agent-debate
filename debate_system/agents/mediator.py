"""Mediator chains — mid-debate summary and final scored verdict."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from agents.utils import format_claim_ledger, format_summaries, format_transcript
from config import MAX_SUMMARY_TOKENS, MAX_VERDICT_TOKENS
from llm.models import get_chat_model
from models.outputs import FinalVerdictOutput, MediatorSummaryOutput
from models.schemas import (
    AgentPersona,
    Citation,
    ClaimScore,
    DebateState,
    DebateTurn,
    FinalVerdict,
    MediatorSummary,
)
from prompts.mediator_prompt import (
    MEDIATOR_SUMMARY_USER,
    MEDIATOR_SYSTEM,
    MEDIATOR_VERDICT_USER,
)
from search.tavily import get_searcher, to_citations

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", MEDIATOR_SYSTEM), ("human", MEDIATOR_SUMMARY_USER)]
)
VERDICT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", MEDIATOR_SYSTEM), ("human", MEDIATOR_VERDICT_USER)]
)


def summary_chain() -> Runnable:
    model = get_chat_model(max_tokens=MAX_SUMMARY_TOKENS)
    return SUMMARY_PROMPT | model.with_structured_output(MediatorSummaryOutput)


def verdict_chain() -> Runnable:
    model = get_chat_model(max_tokens=MAX_VERDICT_TOKENS)
    return VERDICT_PROMPT | model.with_structured_output(FinalVerdictOutput)


def _persona_vars(persona: AgentPersona) -> dict:
    return {
        "name": persona.name,
        "background": persona.background,
        "personality": persona.personality,
        "core_argument_angle": persona.core_argument_angle,
    }


def build_mediator_summary(
    persona: AgentPersona,
    state: DebateState,
    turn_number: int,
) -> tuple[DebateTurn, MediatorSummary]:
    """Mid-debate checkpoint. Returns (turn for the record, structured summary)."""
    search_query = f"{state.topic} research evidence"
    results, search_context = get_searcher().retrieve(search_query)

    out: MediatorSummaryOutput = summary_chain().invoke(
        {
            **_persona_vars(persona),
            "topic": state.topic,
            "transcript": format_transcript(state.turns),
            "claim_ledger": format_claim_ledger(state.claim_ledger),
            "search_results": search_context,
        }
    )

    summary = MediatorSummary(
        turn_number=turn_number,
        state_of_debate=out.state_of_debate,
        contested_ground=out.contested_ground,
        weakest_arguments=out.weakest_arguments,
        probing_question=out.probing_question,
    )

    turn = DebateTurn(
        turn_number=turn_number,
        agent_name=persona.name,
        agent_role=persona.specialty,
        side="mediator",
        thinking=out.thinking,
        argument=(
            f"STATE OF DEBATE: {summary.state_of_debate}\n\n"
            f"CONTESTED GROUND: {summary.contested_ground}\n\n"
            f"WEAKEST ARGUMENTS: {summary.weakest_arguments}\n\n"
            f"PROBING QUESTION: {summary.probing_question}"
        ),
        new_claims=[],
        rebuttal_targets=[],
        confidence=1.0,
        retrieved_sources=to_citations(results),
        search_query=search_query,
    )
    return turn, summary


def build_final_verdict(persona: AgentPersona, state: DebateState) -> FinalVerdict:
    """Score the whole debate."""
    search_query = f"{state.topic} expert analysis conclusion"
    results, search_context = get_searcher().retrieve(search_query)

    out: FinalVerdictOutput = verdict_chain().invoke(
        {
            **_persona_vars(persona),
            "topic": state.topic,
            "transcript": format_transcript(state.turns),
            "claim_ledger": format_claim_ledger(state.claim_ledger),
            "summaries": format_summaries(state.mediator_summaries),
            "search_results": search_context,
        }
    )

    return FinalVerdict(
        argument_scorecard=[ClaimScore(**s.model_dump()) for s in out.argument_scorecard],
        strongest_pro_arguments=out.strongest_pro_arguments,
        strongest_con_arguments=out.strongest_con_arguments,
        fatal_pro_weaknesses=out.fatal_pro_weaknesses,
        fatal_con_weaknesses=out.fatal_con_weaknesses,
        confidence_trajectory=out.confidence_trajectory,
        nuanced_conclusion=out.nuanced_conclusion,
        verdict=out.verdict,
        citations=[Citation(**c.model_dump()) for c in out.citations],
        retrieved_sources=to_citations(results),
        search_query=search_query,
    )
