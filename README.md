# Multi-Agent Debate System

A Python terminal application where AI agents with distinct specialist personas debate any topic you provide. Orchestration runs on **LangGraph** — a stateful, cyclic graph with an LLM supervisor deciding who speaks next — with **LangChain** structured output over **Anthropic Claude**, **OpenAI GPT**, or **Google Gemini** as interchangeable backends, and a color-coded terminal UI built using the `rich` library.

Agents can **search the internet in real time** (via [Tavily](https://tavily.com)) and cite their sources — every argument is grounded in live evidence, not just training data.

You can also **jump into the debate yourself** — join as a human participant on the pro or con side and argue alongside the AI specialists.

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
2. Asks if you'd like to **join the debate yourself** as an extra pro or con voice
3. Runs the debate with alternating turns, each agent making new claims and rebutting prior ones
4. **Fetches live web evidence via Tavily before each turn**, so AI arguments are grounded in real sources
5. Inserts a neutral **Mediator** periodically to assess the state of the debate and ask probing questions
6. Delivers a scored **Final Verdict** at the end, also backed by cited sources
7. Saves the full transcript (including citations) to a `.json` file

---

## Agents

| Agent | Role |
|-------|------|
| **Topic Decomposer** | Runs once at startup to generate all personas dynamically |
| **PRO Specialists** | One expert per debate dimension, arguing in favor |
| **CON Specialists** | One expert per debate dimension, arguing against |
| **Human Participant** *(optional)* | You, arguing on the pro or con side alongside the AI specialists |
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
│                                                                           │
│ 🔗 SOURCES                                                                │
│   [1] NEJM — UBI pilot outcomes in healthcare utilization                 │
│       https://nejm.org/...                                                │
│       "Participants showed a 14% reduction in ER visits..."               │
╰───────────────────────────────────────────────────────────────────────────╯
```

- **Green panels** — PRO side arguments
- **Red panels** — CON side arguments
- **Yellow panels** — Mediator analysis
- **Claim ledger** — live table showing all claims, marked `STANDING` or `REBUTTED`
- **Thinking blocks** — each agent's internal reasoning, surfaced in dim italic (all providers)
- **Confidence bar** — each agent self-rates their argument quality
- **Sources block** — clickable URLs for every source cited in the turn

---

## Human-in-the-Loop

After the AI specialists are generated and confirmed, you're asked:

```
Join the debate yourself as a human participant? [y/N]: y
Which side will you argue? [pro/con]: con
Your name (shown in the transcript) [default: You]: Akash
A short label for your angle/expertise (optional) [default: Human Participant]: Skeptic
```

Say yes and you're added as an **extra** pro or con voice — the AI lineup isn't reduced to make room for you, so sides can end up uneven (that's fine, the turn order handles it).

When it's your turn, the terminal shows you the same context an AI specialist gets — the full transcript so far, the claim ledger (which claims are still `STANDING` and open to rebut), and the mediator's last summary — then prompts you to type your argument (press Enter on an empty line to submit).

Your turn is intentionally lightweight compared to the AI specialists':
- Your argument is registered as a new claim automatically (auto-assigned the next `claim_id`)
- No internal "thinking" step is shown or asked for — that's just for the AI agents
- No rebuttal-target picking or citation prompts — you just argue in your own words

Everything else — turn order, reactive routing, mediator check-ins, the final scored verdict, and the saved transcript — treats your turns exactly like any other specialist's.

---

## Setup

### 1. Prerequisites

- Python 3.11+
- An API key for your chosen LLM provider (see [Supported Providers](#supported-providers))
- *(Optional)* A [Tavily](https://app.tavily.com) API key for live web search

### 2. Install dependencies

```bash
cd debate_system
pip install -r requirements.txt
```

> The requirements file lists all three provider integrations (`langchain-anthropic`, `langchain-openai`, `langchain-google-genai`). Install only the one you need.

### 3. Configure your provider

```bash
cp .env.example .env
```

Open `.env` and set your provider and API keys:

```env
# Pick one: anthropic | openai | gemini
LLM_PROVIDER=anthropic

ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Optional — enables live web search and source citations
TAVILY_API_KEY=tvly-...
```

### 4. Run

```bash
python main.py
```

---

## Web Search & Citations (Tavily)

When `TAVILY_API_KEY` is set, every specialist and mediator call:

1. **Searches the web** before generating its argument, using a domain-targeted query (e.g. `"Universal Basic Income Healthcare Economics"`)
2. **Receives the top results** (title, URL, content snippet) as context injected into the prompt
3. **Cites its sources** as structured output — the UI renders these as a `🔗 SOURCES` block under each argument
4. **Saves citations** in the JSON transcript alongside the full debate

The mediator uses separate searches for its mid-debate summaries (`"{topic} research evidence"`) and its final verdict (`"{topic} expert analysis conclusion"`).

**Without a Tavily key**, the system runs exactly as before — agents argue from model training knowledge with no citations, and no errors are raised.

### Tavily env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `TAVILY_API_KEY` | *(unset)* | API key from [app.tavily.com](https://app.tavily.com). Required to enable search. |
| `SEARCH_ENABLED` | `true` | Set to `false` to disable search without removing the key |
| `TAVILY_MAX_RESULTS` | `3` | Number of search results fetched per agent turn |

---

## Supported Providers

| `LLM_PROVIDER` | API Key Env Var | Default Model | LangChain package |
|----------------|-----------------|---------------|-------------------|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | `langchain-anthropic` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o` | `langchain-openai` |
| `gemini` | `GOOGLE_API_KEY` | `gemini-flash-lite-latest` | `langchain-google-genai` |

Any provider works as long as it supports tool calling — that's what `with_structured_output` uses to guarantee the response shape. Reasoning is captured as a `thinking` field on every schema, so all three providers surface it identically.

### Model overrides

Override the default model for any provider via environment variables:

```env
ANTHROPIC_MODEL=claude-sonnet-4-6
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-flash-lite-latest
```

---

## Configuration

Edit [`config.py`](debate_system/config.py) to tune the system (all token/round defaults live here):

| Constant | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Default Anthropic model |
| `OPENAI_MODEL` | `gpt-4o` | Default OpenAI model |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Default Gemini model |
| `MODEL_TEMPERATURE` | `0.7` | Sampling temperature (routing calls always use 0.0) |
| `DEFAULT_ROUNDS` | `6` | Default number of debate rounds |
| `DEFAULT_SPECIALISTS` | `2` | Default number of specialist pairs |
| `MEDIATOR_SUMMARY_DIVISOR` | `3` | Roughly how many times the mediator interjects per debate |
| `WARN_COMPLEXITY_THRESHOLD` | `20` | Warns if total turns exceed this |
| `GRAPH_RECURSION_LIMIT` | `250` | Max node visits per run (~2 per turn) |
| `CHECKPOINT_DB` | `memory` | `memory`, or a `.sqlite` path for resumable debates |
| `MAX_RETRIES` | `3` | Model retry attempts with exponential backoff |
| `TAVILY_MAX_RESULTS` | `3` | Search results fetched per agent turn |

---

## Project Structure

```
debate_system/
├── main.py                  # Entry point — collects params, starts the graph
├── config.py                # Global constants, model defaults, graph settings
├── requirements.txt
├── .env.example
│
├── graph/                   # ← the orchestration layer
│   ├── state.py             # DebateGraphState channels + claim-ledger reducer
│   ├── nodes.py             # Every debate step as a pure state transition
│   └── builder.py           # StateGraph wiring, conditional edges, checkpointer
│
├── llm/
│   └── models.py            # init_chat_model factory (anthropic | openai | gemini)
│
├── agents/                  # LCEL chains: prompt | model.with_structured_output(...)
│   ├── decomposer.py        # Generates all agent personas from the topic
│   ├── specialist.py        # Builds one debate turn + its claim-ledger delta
│   ├── mediator.py          # Mid-debate summaries and final verdict
│   ├── supervisor.py        # Router chain — which specialist answers next
│   ├── human.py             # Wraps a human's typed argument as a DebateTurn
│   └── utils.py             # Shared transcript/claim formatting helpers
│
├── debate/
│   └── runner.py            # CLI driver: streams graph updates, answers interrupts
│
├── models/
│   ├── schemas.py           # Domain models: Citation, Claim, DebateTurn, AgentPersona…
│   └── outputs.py           # Structured-output schemas bound to the models
│
├── prompts/
│   ├── decomposer_prompt.py
│   ├── specialist_prompt.py # Specialist turn + supervisor routing prompts
│   └── mediator_prompt.py   # Summary + verdict prompts
│
├── search/
│   └── tavily.py            # LangChain Tavily tool wrapper (degrades to no-op)
│
├── ui/
│   └── display.py           # All Rich terminal rendering (panels, ledger, citations, verdict)
│
├── tests/
│   └── test_graph_smoke.py  # Full debate against fake models — no API calls, ~1s
│
└── transcripts/             # Auto-created — JSON debate transcripts saved here
```

---

## Architecture — the debate graph

There is no debate loop anywhere in this codebase. The debate is a **LangGraph `StateGraph`** with a cycle in it, and the shape of any given debate is emergent rather than scripted.

```
START ─> decompose ─> approve ─┬─ regenerate ─> decompose   (loop)
                               ├─ abort ──────> END
                               └─ approve ────> supervisor
                                                    │
    ┌───────────────────────────────────────────────┤
    │  supervisor ─┬─ specialist ─┐                 │
    │              ├─ human ──────┤ back to supervisor
    │              ├─ mediator ───┘                 │
    │              └─ verdict ────> persist ─> END  │
    └───────────────────────────────────────────────┘
```

**State.** One `DebateGraphState` TypedDict holds everything. Three channels accumulate rather than overwrite: `turns` and `mediator_summaries` via `operator.add`, and `claim_ledger` via a custom reducer that *upserts* on `claim_id`. A speaking node returns new claims **and** "rebuttal patches" — stub copies of existing claims carrying only a new `rebutted_by` entry — and the reducer folds them in. No node ever read-modify-writes the whole ledger, which is what would make parallel debate branches safe to add later.

**The supervisor is the only node that decides anything.** It runs after every single turn and re-derives the situation from current state: has the round rolled over, is the debate finished, is a mediator check-in due, which side is owed a reply. The deterministic parts stay in code; the one genuine judgement call — *which* of the eligible same-side specialists can best contest what was just said — is a small structured-output chain (`agents/supervisor.py`). Routing to the next node happens through `add_conditional_edges`.

**Human turns are interrupts, not `input()` calls.** The human node calls LangGraph's `interrupt()`, which suspends the graph mid-run; the driver resumes it with `Command(resume={"argument": ...})`. Persona approval works the same way. Because a node that interrupts re-executes from the top on resume, the expensive `decompose` call lives in its own node — approving the cast can never silently regenerate it.

**Checkpointing** is what makes any of that work: state has to be durable across the pause. `InMemorySaver` by default; set `CHECKPOINT_DB=debate.sqlite` and a debate survives the process dying and can be resumed on the same `thread_id`.

**Nothing in `graph/` knows about the terminal.** `debate/runner.py` consumes `graph.stream(..., stream_mode="updates")` and renders each node update with `rich`. Swapping that one file for a web handler is the entire cost of turning this into a service.

---

## How the Debate Works

1. **Decomposition** — A single structured-output call analyzes the topic and generates K specialist pairs + a mediator, each with a name, specialty, background, and distinct personality.

2. **Turn Order** — The supervisor node alternates PRO/CON and gives everyone exactly one turn per round. When more than one specialist on the owed side is still eligible, a routing chain picks whichever is best positioned to answer the previous argument, making the debate feel reactive. If you joined as a human participant, you're in this rotation like any other specialist.

3. **Live Search** — Before generating each turn, the agent searches Tavily for domain-relevant evidence (e.g. `"UBI healthcare economics"`). Results are injected into the prompt as numbered source blocks. The agent weaves this evidence into its argument and explicitly cites what it used.

4. **Claim Ledger** — Every turn must register at least one new `Claim` and rebut at least one `[STANDING]` claim from the opposing side. This prevents parallel monologues. `claim_id`s are assigned by the system, not the model — models are unreliable at continuing a sequence, and a collision would silently corrupt the ledger. Rebuttal targets are validated against the ledger and dropped if they point at the agent's own side.

5. **Thinking** — Every agent reasons before it speaks, and that reasoning is rendered above the argument. It's carried as a `thinking` field on the structured output rather than via provider-specific extended-thinking APIs (which conflict with the forced tool-use that powers structured output) — so it behaves identically across all three providers.

6. **Mediator Summaries** — Roughly three times per debate, the supervisor routes to the mediator instead of a specialist for a structured analysis: state of debate, contested ground, weakest arguments, and a probing question. The mediator also searches for current evidence to inform its assessment.

7. **Final Verdict** — When the last round completes, the supervisor routes to the verdict node. The mediator scores every claim (1–5), identifies fatal weaknesses on each side, tracks confidence trajectories, and delivers a nuanced conclusion with its own cited sources.

8. **Transcript** — The full debate is saved to `transcripts/debate_transcript_<timestamp>.json`, including all citations. If you press `Ctrl+C` mid-debate, the driver reads the last checkpoint and saves a partial transcript before exiting.

---

## Testing

A full debate — persona approval, the regenerate loop, 15 turns, human interrupts, mediator check-ins, verdict, transcript — runs against fake chat models in about a second, with no API keys and no network:

```bash
cd debate_system
python tests/test_graph_smoke.py
```

It asserts the things that are actually easy to break: the ledger reducer stays pure, every specialist speaks exactly once per round, sides alternate, rebuttals only ever cross sides, turn numbers stay strictly increasing, and the saved JSON matches final state.

---

## Example Topics to Try

- *"Nuclear energy is essential for meeting climate goals"*
- *"Social media does more harm than good"*
- *"Artificial intelligence will eliminate more jobs than it creates"*
- *"Universal Basic Income should be implemented globally"*
- *"Gene editing in humans should be permitted for non-medical enhancement"*
