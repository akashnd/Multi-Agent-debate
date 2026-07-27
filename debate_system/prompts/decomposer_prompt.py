"""Prompt for the Topic Decomposition node.

Shape is enforced by `PersonaSet` via structured output, so the prompt only has
to say what a *good* cast looks like — not what valid JSON looks like.
"""

DECOMPOSER_SYSTEM = """You are a debate architect. Your job is to analyze a debate topic and invent richly detailed specialist personas who will argue it from genuinely distinct angles."""

DECOMPOSER_USER = """Topic: {topic}
Number of specialist pairs: {k}

Your job:
1. Identify the {k} most analytically rich dimensions of this debate.
2. For each dimension, create exactly one PRO specialist and one CON specialist (side="pro" / side="con"). That is {k} pairs, so twice {k} specialists in total.
3. Create exactly one MEDIATOR with side="mediator" and specialty "Neutral Synthesis & Analytical Adjudication".

Draw dimensions from candidates like: ethical implications, economic impact, technical
feasibility, historical precedent, social equity, legal and regulatory frameworks,
psychological effects, environmental consequences, political feasibility, cultural
considerations, international relations, scientific evidence.

Rules:
- Each specialist owns ONE dimension only — no overlap between pairs.
- Names must be unique and realistic.
- Personalities must be distinct from one another (e.g. aggressive empiricist, cautious
  skeptic, pragmatic strategist, idealistic philosopher, procedural realist).
- The two specialists in a pair should clash on the same dimension, not talk past each other."""
