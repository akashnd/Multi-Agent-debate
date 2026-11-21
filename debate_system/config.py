"""Global configuration constants for the Multi-Agent Debate System."""

import os

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Set LLM_PROVIDER in your .env file: anthropic | openai | gemini
# Default is anthropic. Resolved into a LangChain chat model in llm/models.py.

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.7"))

# ── Token limits (per structured-output call) ─────────────────────────────────
MAX_ARGUMENT_TOKENS = 8000  # specialist turns
MAX_VERDICT_TOKENS = 16000  # final verdict
MAX_SUMMARY_TOKENS = 4000  # mediator mid-debate summaries
MAX_PERSONA_TOKENS = 4096  # topic decomposition
MAX_ROUTER_TOKENS = 512  # supervisor's next-speaker decision

# ── Debate structure ──────────────────────────────────────────────────────────
MEDIATOR_SUMMARY_DIVISOR = 3  # mediator summarises ~this many times per debate
DEFAULT_SPECIALISTS = 2  # default K specialist pairs
DEFAULT_ROUNDS = 6  # default N debate rounds
WARN_COMPLEXITY_THRESHOLD = 20  # warn user if total turns exceed this

# Backwards-compatible alias (older code referred to this name)
MEDIATOR_SUMMARY_INTERVAL = MEDIATOR_SUMMARY_DIVISOR

# ── Graph execution ───────────────────────────────────────────────────────────
# Every node visit counts against the recursion limit, so a long debate needs a
# generous ceiling: ~3 node visits per turn (supervisor -> speaker -> supervisor).
GRAPH_RECURSION_LIMIT = int(os.environ.get("GRAPH_RECURSION_LIMIT", "250"))

# Where the LangGraph checkpointer persists state.
#   unset / "memory"  -> InMemorySaver (state lives for the process only)
#   <path>.sqlite     -> SqliteSaver   (debate survives a crash; resumable)
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "memory")

# ── Resilience ────────────────────────────────────────────────────────────────
MAX_RETRIES = 3  # LangChain-level retry attempts per model call
RETRY_BASE_DELAY = 2.0  # base delay (s) for exponential backoff
STREAM_DELAY = 0.01  # seconds between streamed tokens (display pacing)
TURN_TIMEOUT = 120  # seconds before showing thinking spinner

# ── Tavily Web Search ─────────────────────────────────────────────────────────
# Set TAVILY_API_KEY in your .env to enable evidence-grounded arguments.
# SEARCH_ENABLED defaults to true; set to "false" to disable without removing the key.
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "3"))
