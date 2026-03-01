# Multi-Agent Debate System

A Python terminal application where AI agents with distinct specialist personas debate any topic you provide. Supports **Anthropic Claude**, **OpenAI GPT**, and **Google Gemini** as interchangeable backends, with a rich color-coded terminal UI built using the `rich` library.

---

## What It Does

You give it a topic. It generates a cast of specialist debaters — each with a unique name, background, and intellectual personality — and runs a structured debate between them in your terminal, live.

```
Using LLM provider: anthropic

Enter debate topic: Should Universal Basic Income be implemented globally?
Number of debate rounds [default 6]: 4
Number of specialist pairs [default 2]: 3
```

The system then:
1. Generates PRO and CON specialists across the most relevant dimensions of the topic (economic, ethical, political, etc.)
2. Runs the debate with alternating turns, each agent making new claims and rebutting prior ones
3. Inserts a neutral **Mediator** periodically to assess the state of the debate and ask probing questions
4. Delivers a scored **Final Verdict** at the end
5. Saves the full transcript to a `.json` file

---

## Agents

| Agent | Role |
|-------|------|
| **Topic Decomposer** | Runs once at startup to generate all personas dynamically |
| **PRO Specialists** | One expert per debate dimension, arguing in favor |
| **CON Specialists** | One expert per debate dimension, arguing against |
| **Mediator** | Tracks the debate, intervenes with analysis, delivers final verdict |

---

## Terminal UI Preview

```
╭─ Turn 4 | DR. SARAH CHEN | Healthcare Economics | PRO ──────────────────╮
│ 💭 THINKING (internal)                                                    │
│   [dim] She's been weak on cost data. I should hit the McKinsey           │
│   numbers hard and force a specific rebuttal...                           │
│                                                                           │
│ 🗣 ARGUMENT                                                               │
│   The economic case is unambiguous. Hospital readmission rates fell...    │
│                                                                           │
│ 🎯 REBUTTING: claim_003 (Dr. Marcus Webb's patient safety claim)         │
│ 📊 Confidence: ████████░░ 0.82                                           │
╰───────────────────────────────────────────────────────────────────────────╯
```

- **Green panels** — PRO side arguments
- **Red panels** — CON side arguments
- **Yellow panels** — Mediator analysis
- **Claim ledger** — live table showing all claims, marked `STANDING` or `REBUTTED`
- **Thinking blocks** — internal reasoning surfaced in dim italic (Anthropic and Gemini thinking models)
- **Confidence bar** — each agent self-rates their argument quality

---

## Setup

### 1. Prerequisites

- Python 3.11+
- An API key for your chosen provider (see [Supported Providers](#supported-providers))

### 2. Install dependencies

```bash
cd debate_system
pip install -r requirements.txt
```

> The requirements file includes `openai` and `google-generativeai`. If you only want one provider, you can install just the packages you need.

### 3. Configure your provider

```bash
cp .env.example .env
```

Open `.env` and set your provider and API key:

```env
# Pick one: anthropic | openai | gemini
LLM_PROVIDER=anthropic

ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

### 4. Run

```bash
python main.py
```

---

## Supported Providers

| `LLM_PROVIDER` | API Key Env Var | Default Model | Extended Thinking |
|----------------|-----------------|---------------|-------------------|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | Native — surfaces internal reasoning |
| `openai` | `OPENAI_API_KEY` | `gpt-4o` | Not exposed — o-series reasons internally |
| `gemini` | `GOOGLE_API_KEY` | `gemini-1.5-pro` | Available on thinking-capable models |

### Model overrides

Override the default model for any provider via environment variables:

```env
ANTHROPIC_MODEL=claude-sonnet-4-6
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-1.5-pro
```

**Gemini thinking models** (set `GEMINI_MODEL` to one of these to enable thought extraction):
- `gemini-2.0-flash-thinking-exp`
- `gemini-2.5-pro-preview-03-25`

**OpenAI o-series models** (reasoning happens internally, not surfaced):
- `o1`, `o3-mini`, `o4-mini`, etc. — set `OPENAI_MODEL` accordingly

---

## Configuration

Edit [`config.py`](config.py) to tune the system (all token/round defaults live here):

| Constant | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Default Anthropic model (env override: `ANTHROPIC_MODEL`) |
| `OPENAI_MODEL` | `gpt-4o` | Default OpenAI model (env override: `OPENAI_MODEL`) |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Default Gemini model (env override: `GEMINI_MODEL`) |
| `THINKING_BUDGET` | `3000` | Token budget for extended thinking (Anthropic / Gemini thinking models) |
| `DEFAULT_ROUNDS` | `6` | Default number of debate rounds |
| `DEFAULT_SPECIALISTS` | `2` | Default number of specialist pairs |
| `MEDIATOR_SUMMARY_INTERVAL` | `3` | How often the mediator intervenes (every N turns) |
| `WARN_COMPLEXITY_THRESHOLD` | `20` | Warns if total turns exceed this |
| `MAX_RETRIES` | `3` | API retry attempts with exponential backoff |

---

## Project Structure

```
debate_system/
├── main.py                  # Entry point — CLI prompts and orchestration
├── config.py                # Global constants and model defaults
├── requirements.txt
├── .env.example
│
├── llm/                     # Provider abstraction layer
│   ├── base.py              # Abstract LLMClient interface
│   ├── anthropic_client.py  # Anthropic implementation (extended thinking)
│   ├── openai_client.py     # OpenAI implementation (GPT-4o + o-series)
│   ├── gemini_client.py     # Google Gemini implementation (thinking models)
│   └── factory.py           # Reads LLM_PROVIDER env var and returns the right client
│
├── agents/
│   ├── decomposer.py        # Generates all agent personas from the topic
│   ├── specialist.py        # Runs one debate turn for a specialist
│   ├── mediator.py          # Mid-debate summaries and final verdict
│   └── utils.py             # Shared transcript/claim formatting helpers
│
├── debate/
│   ├── orchestrator.py      # Debate loop, claim ledger updates, transcript saving
│   └── turn_selector.py     # Dynamic pro/con alternation and mediator scheduling
│
├── models/
│   └── schemas.py           # Pydantic models: Claim, DebateTurn, AgentPersona, etc.
│
├── prompts/
│   ├── decomposer_prompt.py
│   ├── specialist_prompt.py
│   └── mediator_prompt.py
│
├── ui/
│   └── display.py           # All Rich terminal rendering
│
└── transcripts/             # Auto-created — JSON debate transcripts saved here
```

---

## How the Debate Works

1. **Decomposition** — A single LLM call analyzes the topic and generates K specialist pairs + a mediator, each with a name, specialty, background, and distinct personality.

2. **Turn Order** — Each round alternates PRO/CON speakers. After each turn, a lightweight routing call picks which opposing specialist is best positioned to respond, making the debate feel reactive.

3. **Claim Ledger** — Every turn must register at least one new `Claim` (with a unique `claim_id`) and rebut at least one `[STANDING]` claim from the opposing side. This prevents parallel monologues.

4. **Extended Thinking** — When supported by the provider, internal reasoning is surfaced before each spoken argument. On Anthropic, this uses the native `thinking` parameter. On Gemini thinking models, thought blocks are extracted from the response parts. On standard OpenAI models, reasoning is not exposed.

5. **Mediator Summaries** — Every `total_turns // 3` turns, the mediator intervenes with a structured analysis: state of debate, contested ground, weakest arguments, and a probing question.

6. **Final Verdict** — After all turns, the mediator scores every claim (1–5), identifies fatal weaknesses on each side, tracks confidence trajectories, and delivers a nuanced conclusion.

7. **Transcript** — The full debate is saved to `transcripts/debate_transcript_<timestamp>.json`. If you press `Ctrl+C` mid-debate, a partial transcript is saved before exiting.

---

## Example Topics to Try

- *"Nuclear energy is essential for meeting climate goals"*
- *"Social media does more harm than good"*
- *"Artificial intelligence will eliminate more jobs than it creates"*
- *"Universal Basic Income should be implemented globally"*
- *"Gene editing in humans should be permitted for non-medical enhancement"*
