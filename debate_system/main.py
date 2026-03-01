"""Multi-Agent Debate System — Entry Point."""

import sys
import os
from pathlib import Path

# Add project root to path so modules resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from config import DEFAULT_SPECIALISTS, DEFAULT_ROUNDS, WARN_COMPLEXITY_THRESHOLD
from llm.factory import create_client
from agents.decomposer import TopicDecomposer
from debate.orchestrator import DebateOrchestrator
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


def main():
    load_dotenv()

    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    console.print(f"[dim]Using LLM provider: [bold]{provider}[/bold][/dim]")

    # create_client() validates the API key and exits with a clear message if missing
    try:
        client = create_client()
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[bold red]Failed to initialise LLM client:[/bold red] {e}")
        sys.exit(1)

    display = DebateDisplay()
    display.show_welcome()

    # ── Get debate parameters ──────────────────────────────────────────────
    topic = input("Enter debate topic: ").strip()
    if not topic:
        console.print("[yellow]No topic entered. Using default: 'Social media does more harm than good'[/yellow]")
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
        confirm = input("Continue anyway? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[dim]Exiting.[/dim]")
            sys.exit(0)

    # ── Generate agents ────────────────────────────────────────────────────
    console.print(f"\n[dim]Generating {k_specialists * 2 + 1} specialist agents...[/dim]")
    decomposer = TopicDecomposer(client)

    try:
        agents = decomposer.decompose(topic, k_specialists)
    except Exception as e:
        display.show_error(f"Failed to generate agents: {e}")
        sys.exit(1)

    # Display generated agents and let user confirm or regenerate
    while True:
        display.show_agents_generated(agents)

        for agent in agents:
            display.show_agent_details(agent)

        confirm = input("Proceed with these agents? [Y/n/r (regenerate)]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            break
        elif confirm in ("r", "regen", "regenerate"):
            console.print("[dim]Regenerating agents...[/dim]")
            try:
                agents = decomposer.decompose(topic, k_specialists)
            except Exception as e:
                display.show_error(f"Failed to regenerate agents: {e}")
                sys.exit(1)
        else:
            console.print("[dim]Exiting.[/dim]")
            sys.exit(0)

    # ── Run the debate ─────────────────────────────────────────────────────
    try:
        orchestrator = DebateOrchestrator(
            client=client,
            topic=topic,
            n_rounds=n_rounds,
            agents=agents,
            display=display,
        )
        orchestrator.run()
    except SystemExit:
        pass
    except Exception as e:
        display.show_error(f"Unexpected error during debate: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
