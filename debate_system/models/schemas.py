"""Pydantic models for the Multi-Agent Debate System."""

from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    title: str
    url: str
    snippet: str = ""        # key quote or summary the agent drew on


class Claim(BaseModel):
    claim_id: str                    # unique id e.g. "claim_001"
    agent_name: str                  # who made the claim
    side: str                        # "pro" | "con"
    content: str                     # the actual claim text
    turn_number: int
    rebutted_by: list[str] = Field(default_factory=list)  # list of claim_ids that rebut this


class DebateTurn(BaseModel):
    turn_number: int
    agent_name: str
    agent_role: str                  # their specialty
    side: str                        # "pro" | "con" | "mediator"
    thinking: str                    # internal reasoning (from Claude's extended thinking)
    argument: str                    # the actual spoken argument
    new_claims: list[Claim] = Field(default_factory=list)
    rebuttal_targets: list[str] = Field(default_factory=list)  # claim_ids this turn rebuts
    confidence: float = 0.5          # 0.0 to 1.0, agent's self-rated confidence this turn
    citations: list[Citation] = Field(default_factory=list)    # sources cited this turn


class AgentPersona(BaseModel):
    name: str
    side: str                        # "pro" | "con" | "mediator"
    specialty: str                   # e.g. "Healthcare Economics"
    background: str                  # 2-3 sentence bio
    personality: str                 # e.g. "Aggressive empiricist who demands data"
    core_argument_angle: str         # their unique lens on the topic


class MediatorSummary(BaseModel):
    turn_number: int
    state_of_debate: str
    contested_ground: str
    weakest_arguments: str
    probing_question: str


class ClaimScore(BaseModel):
    claim_id: str
    claim_summary: str
    defense_rating: int              # 1-5
    successfully_rebutted: bool
    notes: str


class FinalVerdict(BaseModel):
    argument_scorecard: list[ClaimScore]
    strongest_pro_arguments: list[str]
    strongest_con_arguments: list[str]
    fatal_pro_weaknesses: str
    fatal_con_weaknesses: str
    confidence_trajectory: str
    nuanced_conclusion: str
    verdict: str                     # which side had stronger case and why
    citations: list[Citation] = Field(default_factory=list)  # sources supporting the verdict


class DebateState(BaseModel):
    topic: str
    total_turns: int
    current_turn: int = 0
    agents: list[AgentPersona] = Field(default_factory=list)
    turns: list[DebateTurn] = Field(default_factory=list)
    claim_ledger: list[Claim] = Field(default_factory=list)
    mediator_summaries: list[MediatorSummary] = Field(default_factory=list)
