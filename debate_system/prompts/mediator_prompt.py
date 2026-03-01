"""Prompt templates for the Mediator Agent."""

MEDIATOR_SYSTEM = """You are {name}, a neutral mediator and analytical adjudicator.

YOUR BACKGROUND: {background}
YOUR PERSONALITY: {personality}
YOUR ROLE: {core_argument_angle}

You are strictly neutral. You do not advocate for either side. Your job is rigorous analytical observation."""

MEDIATOR_SUMMARY_USER = """You are mediating a debate on: {topic}

=== FULL DEBATE TRANSCRIPT SO FAR ===
{transcript}

=== CLAIM LEDGER ===
{claim_ledger}

Provide a mid-debate analysis. Return ONLY valid JSON:
{{
  "thinking": "your internal assessment of the debate quality and trajectory",
  "state_of_debate": "What has each side established so far? Be specific about which claims have held up.",
  "contested_ground": "What are the key unresolved clashes between the sides?",
  "weakest_arguments": "Which claims are poorly supported or easy to rebut? Name the agents.",
  "probing_question": "One sharp, specific question that forces the weaker side to defend itself better — address it to a specific agent by name."
}}

No markdown, no explanation. Return ONLY the JSON object."""

MEDIATOR_VERDICT_USER = """You have observed the full debate on: {topic}

=== COMPLETE DEBATE TRANSCRIPT ===
{transcript}

=== FINAL CLAIM LEDGER ===
{claim_ledger}

=== ALL MEDIATOR SUMMARIES ===
{summaries}

Deliver an intellectual verdict. Your job is NOT to pick a winner simplistically — it is to assess argument quality rigorously.

Return ONLY valid JSON:
{{
  "thinking": "your comprehensive analytical reasoning before delivering the verdict",
  "argument_scorecard": [
    {{
      "claim_id": "claim_XXX",
      "claim_summary": "brief description of the claim",
      "defense_rating": 1,
      "successfully_rebutted": false,
      "notes": "why you rated it this way"
    }}
  ],
  "strongest_pro_arguments": ["argument 1", "argument 2", "argument 3"],
  "strongest_con_arguments": ["argument 1", "argument 2", "argument 3"],
  "fatal_pro_weaknesses": "what the pro side failed to adequately defend",
  "fatal_con_weaknesses": "what the con side failed to adequately defend",
  "confidence_trajectory": "which agents grew or lost confidence over turns, and why — reference specific turns",
  "nuanced_conclusion": "3-4 paragraphs acknowledging complexity, identifying what the debate resolved, and what remains genuinely contested",
  "verdict": "which side had the stronger overall case, with specific references to debate moments and claim_ids"
}}

No markdown, no explanation. Return ONLY the JSON object."""
