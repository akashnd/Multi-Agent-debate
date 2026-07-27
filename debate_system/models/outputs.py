"""Structured-output schemas.

These are the shapes we bind to a LangChain chat model with
`model.with_structured_output(Schema)`. The provider is then responsible for
returning something that validates against the schema — which is why there is
no JSON parsing, no code-fence stripping and no "ask the LLM to repair its own
malformed JSON" fallback anywhere in this codebase any more.

Note on `thinking`: rather than relying on provider-specific extended-thinking
APIs (which conflict with forced tool-use, the mechanism behind structured
output), every schema carries reasoning in-band as its first field. Models fill
it before the answer fields, so it still acts as a scratchpad, and it works
identically across Anthropic / OpenAI / Gemini.
"""

from pydantic import BaseModel, Field

from models.schemas import Citation, ClaimScore


class PersonaSpec(BaseModel):
    """One debate participant invented by the decomposer."""

    name: str = Field(description="Unique, realistic full name")
    side: str = Field(description='Exactly one of: "pro", "con", "mediator"')
    specialty: str = Field(description="One specific area of expertise / debate dimension")
    background: str = Field(description="2-3 sentence professional background")
    personality: str = Field(
        description="Distinct intellectual personality, e.g. 'aggressive empiricist who demands data'"
    )
    core_argument_angle: str = Field(description="This persona's unique lens on the topic")


class PersonaSet(BaseModel):
    """The full cast for a debate: K pro/con pairs plus one neutral mediator."""

    thinking: str = Field(description="Which dimensions of the topic you chose, and why")
    agents: list[PersonaSpec] = Field(
        description="2*K specialists: one pro and one con for each debate dimension"
    )
    mediator: PersonaSpec = Field(description="A single neutral mediator with side='mediator'")


class ProposedClaim(BaseModel):
    """A new claim a specialist puts on the record this turn."""

    content: str = Field(description="Precise, self-contained statement of the claim")


class SpecialistTurnOutput(BaseModel):
    """One specialist's contribution to the debate."""

    thinking: str = Field(
        description="Candid internal strategy: what's weak on the other side, what evidence you'll use"
    )
    argument: str = Field(
        description="The spoken argument — substantive, in-character, specific, pointed"
    )
    new_claims: list[ProposedClaim] = Field(
        description="At least one new original claim made this turn"
    )
    rebuttal_targets: list[str] = Field(
        default_factory=list,
        description="claim_ids from the ledger that this turn directly rebuts, e.g. ['claim_003']",
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Self-rated confidence in this turn's argument"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Search results actually relied on. Empty list if none."
    )


class MediatorSummaryOutput(BaseModel):
    """The mediator's mid-debate analytical checkpoint."""

    thinking: str = Field(description="Your internal assessment of debate quality and trajectory")
    state_of_debate: str = Field(
        description="What each side has established, naming which claims have held up"
    )
    contested_ground: str = Field(description="The key unresolved clashes between the sides")
    weakest_arguments: str = Field(
        description="Which claims are poorly supported or easy to rebut — name the agents"
    )
    probing_question: str = Field(
        description="One sharp question forcing the weaker side to defend itself, addressed to a named agent"
    )


class FinalVerdictOutput(BaseModel):
    """The mediator's scored adjudication of the whole debate."""

    thinking: str = Field(description="Comprehensive analytical reasoning before the verdict")
    argument_scorecard: list[ClaimScore] = Field(
        description="One entry per significant claim, with a 1-5 defense_rating"
    )
    strongest_pro_arguments: list[str]
    strongest_con_arguments: list[str]
    fatal_pro_weaknesses: str = Field(description="What the pro side failed to adequately defend")
    fatal_con_weaknesses: str = Field(description="What the con side failed to adequately defend")
    confidence_trajectory: str = Field(
        description="Which agents gained or lost confidence across turns, referencing specific turns"
    )
    nuanced_conclusion: str = Field(
        description="3-4 paragraphs on what the debate resolved and what remains contested"
    )
    verdict: str = Field(
        description="Which side had the stronger case, citing specific moments and claim_ids"
    )
    citations: list[Citation] = Field(default_factory=list)


class SpeakerChoice(BaseModel):
    """The supervisor's routing decision: who answers the previous speaker."""

    speaker: str = Field(description="Name of the specialist who should speak next — name only")
    reason: str = Field(description="One sentence on why they are the best responder")
