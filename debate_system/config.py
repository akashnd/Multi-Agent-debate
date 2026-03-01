"""Global configuration constants for the Multi-Agent Debate System."""

import os

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Set LLM_PROVIDER in your .env file: anthropic | openai | gemini
# Default is anthropic.

# Per-provider model defaults (override via env vars)
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL",    "gpt-4o")
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL",    "gemini-flash-lite-latest")

# ── Thinking / Reasoning ──────────────────────────────────────────────────────
# Used by Anthropic extended thinking. Ignored for OpenAI/Gemini non-thinking models.
THINKING_BUDGET = 3000          # token budget for extended thinking per turn

# ── Token limits ──────────────────────────────────────────────────────────────
MAX_ARGUMENT_TOKENS = 8000      # max tokens for specialist turns (includes thinking)
MAX_VERDICT_TOKENS  = 16000     # max tokens for final verdict (includes thinking)
MAX_SUMMARY_TOKENS  = 4000      # max tokens for mediator mid-debate summaries

# ── Debate structure ──────────────────────────────────────────────────────────
MEDIATOR_SUMMARY_INTERVAL  = 3  # every N turns, mediator summarizes
DEFAULT_SPECIALISTS        = 2  # default K specialist pairs
DEFAULT_ROUNDS             = 6  # default N debate rounds
WARN_COMPLEXITY_THRESHOLD  = 20 # warn user if total turns exceed this

# ── Resilience ────────────────────────────────────────────────────────────────
MAX_RETRIES       = 3           # max API retry attempts
RETRY_BASE_DELAY  = 2.0         # base delay (s) for exponential backoff
STREAM_DELAY      = 0.01        # seconds between streamed tokens (display pacing)
TURN_TIMEOUT      = 120         # seconds before showing thinking spinner
