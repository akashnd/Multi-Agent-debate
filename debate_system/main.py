"""Multi-Agent Debate System — Entry Point.

Collects debate parameters, then hands everything to the LangGraph debate graph.
All orchestration (cast generation, turn order, mediator interjections, the
human's turn, the verdict, persistence) happens inside the graph; this file only
gathers inputs and starts it.
"""

import os
import sys
from pathlib import Path

# Add project root to path so modules resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from config import DEFAULT_ROUNDS, DEFAULT_SPECIALISTS, WARN_COMPLEXITY_THRESHOLD
from debate.runner import DebateRunner
from models.schemas import AgentPersona
from ui.display import DebateDisplay, console


def get_int_input(prompt: str, default: int, min_val: int = 1, max_val: int = 10) -> int:
    """Prompt the user for an integer with a default value."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            console.print(f"[yellow]Please enter a number between {min_val} and {max_val}.[/yellow]")
        except ValueError:
            console.print("[yellow]Please enter a valid number.[/yellow]")


def maybe_human_participant() -> AgentPersona | None:
    """Ask whether a human wants to join the debate as an extra pro/con voice."""
    join = input("\nJoin the debate yourself as a human participant? [y/N]: ").strip().lower()
    if join not in ("y", "yes"):
        return None

    side = ""
    while side not in ("pro", "con"):
        side = input("Which side will you argue? [pro/con]: ").strip().lower()

    name = input("Your name (shown in the transcript) [default: You]: ").strip() or "You"
    specialty = (
        input("A short label for your angle/expertise (optional) [default: Human Participant]: ").strip()
        or "Human Participant"
    )

    console.print(f"\n[bold]{name}[/bold] will join the {side.upper()} side.")
    return AgentPersona(
        name=name,
        side=side,
        specialty=specialty,
        background="A human participant arguing their own views.",
        personality="Speaks for themselves.",
        core_argument_angle=specialty,
        is_human=True,
    )


def main():
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    console.print(f"[dim]Using LLM provider: [bold]{provider}[/bold][/dim]")

    display = DebateDisplay()
    display.show_welcome()

    # ── Get debate parameters ──────────────────────────────────────────────
    topic = input("Enter debate topic: ").strip()
    if not topic:
        console.print(
            "[yellow]No topic entered. Using default: 'Social media does more harm than good'[/yellow]"
        )
        topic = "Social media does more harm than good"

    n_rounds = get_int_input(
        f"Number of debate rounds [default {DEFAULT_ROUNDS}]: ",
        default=DEFAULT_ROUNDS,
        min_val=1,
        max_val=20,
    )
    k_specialists = get_int_input(
        f"Number of specialist pairs [default {DEFAULT_SPECIALISTS}]: ",
        default=DEFAULT_SPECIALISTS,
        min_val=1,
        max_val=5,
    )

    # Complexity warning
    total_turns = k_specialists * 2 * n_rounds
    if total_turns > WARN_COMPLEXITY_THRESHOLD:
        console.print(
            f"\n[yellow]Warning:[/yellow] {total_turns} total turns may produce very long "
            f"context windows and high API costs. Consider reducing rounds or specialist pairs."
        )
        if input("Continue anyway? [y/N]: ").strip().lower() not in ("y", "yes"):
            console.print("[dim]Exiting.[/dim]")
            sys.exit(0)

    human_persona = maybe_human_participant()

    # ── Run the debate graph ───────────────────────────────────────────────
    console.print(f"\n[dim]Generating {k_specialists * 2 + 1} specialist agents...[/dim]")

    runner = DebateRunner(display)
    console.print(f"[dim]Graph thread: {runner.thread_id}[/dim]")

    try:
        runner.run(
            topic=topic,
            n_rounds=n_rounds,
            k_specialists=k_specialists,
            human_persona=human_persona,
        )
    except SystemExit:
        raise
    except Exception as e:
        display.show_error(f"Unexpected error during debate: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
