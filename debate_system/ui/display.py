"""Rich terminal UI for the Multi-Agent Debate System."""

from rich import box
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from agents.utils import format_claim_ledger, format_transcript, get_last_mediator_summary
from models.schemas import AgentPersona, Claim, DebateState, DebateTurn, FinalVerdict

# Singleton console — never create multiple Console instances
console = Console()

SIDE_COLORS = {
    "pro": "green",
    "con": "red",
    "mediator": "yellow",
}

SIDE_LABELS = {
    "pro": "PRO",
    "con": "CON",
    "mediator": "MEDIATOR",
}


class DebateDisplay:
    """All terminal rendering for the debate system."""

    def __init__(self):
        self._spinner_live: Live | None = None

    # ─────────────────────────────────────────────
    # Welcome & Setup
    # ─────────────────────────────────────────────

    def show_welcome(self):
        console.print()
        console.print(Rule("[bold cyan]MULTI-AGENT DEBATE SYSTEM[/bold cyan]", style="cyan"))
        console.print()

    def show_agents_generated(self, agents: list[AgentPersona]):
        console.print()
        console.print(Rule("[bold]GENERATED AGENTS[/bold]", style="dim"))
        console.print()

        pro_table = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
        pro_table.add_column("Name", style="green")
        pro_table.add_column("Specialty", style="green")
        pro_table.add_column("Personality")

        con_table = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
        con_table.add_column("Name", style="red")
        con_table.add_column("Specialty", style="red")
        con_table.add_column("Personality")

        mediator_info = None

        for agent in agents:
            display_name = f"{agent.name} [bold](You)[/bold]" if agent.is_human else agent.name
            if agent.side == "pro":
                pro_table.add_row(display_name, agent.specialty, agent.personality)
            elif agent.side == "con":
                con_table.add_row(display_name, agent.specialty, agent.personality)
            elif agent.side == "mediator":
                mediator_info = agent

        console.print(
            Panel(pro_table, title="[bold green]PRO SIDE[/bold green]", border_style="green")
        )
        console.print(Panel(con_table, title="[bold red]CON SIDE[/bold red]", border_style="red"))

        if mediator_info:
            console.print(
                Panel(
                    f"[bold]{mediator_info.name}[/bold]  —  {mediator_info.specialty}\n"
                    f"[dim]{mediator_info.personality}[/dim]",
                    title="[bold yellow]MEDIATOR[/bold yellow]",
                    border_style="yellow",
                )
            )
        console.print()

    def show_agent_details(self, agent: AgentPersona):
        color = SIDE_COLORS.get(agent.side, "white")
        console.print(
            f"  [{color}]{agent.name}[/{color}] — {agent.specialty}\n"
            f"  [dim]{agent.background}[/dim]\n"
            f"  [italic]Angle: {agent.core_argument_angle}[/italic]"
        )
        console.print()

    # ─────────────────────────────────────────────
    # Debate Start
    # ─────────────────────────────────────────────

    def show_debate_start(self, state: DebateState):
        console.print()
        console.print(
            Rule(
                f"[bold cyan]DEBATE BEGINS — {state.total_turns} turns | {state.topic}[/bold cyan]",
                style="cyan",
            )
        )
        console.print()

    # ─────────────────────────────────────────────
    # Thinking Spinner
    # ─────────────────────────────────────────────

    def show_thinking_spinner(self, agent_name: str):
        """Show a spinner while waiting for an agent's response."""
        console.print(f"[dim]  {agent_name} is thinking...[/dim]")

    # ─────────────────────────────────────────────
    # Human Turn
    # ─────────────────────────────────────────────

    def show_human_turn_context(self, agent: AgentPersona, state: DebateState):
        """Show a human participant everything they need before typing their turn."""
        color = SIDE_COLORS.get(agent.side, "white")
        label = SIDE_LABELS.get(agent.side, agent.side.upper())

        console.print()
        console.print(
            Rule(
                f"[bold {color}]YOUR TURN — {agent.name} ({label}) | "
                f"Turn {state.current_turn} of {state.total_turns}[/bold {color}]",
                style=color,
            )
        )
        console.print()

        console.print(
            Panel(
                format_transcript(state.turns),
                title="[bold]Debate So Far[/bold]",
                border_style="dim",
            )
        )
        console.print(
            Panel(
                format_claim_ledger(state.claim_ledger),
                title=(
                    "[bold]Claim Ledger[/bold] [dim](standing claims are still open to rebut)[/dim]"
                ),
                border_style="dim",
            )
        )

        summary = get_last_mediator_summary(state)
        if "No mediator summary" not in summary:
            console.print(
                Panel(
                    summary,
                    title="[bold yellow]Mediator's Last Summary[/bold yellow]",
                    border_style="yellow",
                )
            )

        console.print()

    # ─────────────────────────────────────────────
    # Specialist Turn
    # ─────────────────────────────────────────────

    def show_specialist_turn(self, turn: DebateTurn, state: DebateState):
        color = SIDE_COLORS[turn.side]
        label = SIDE_LABELS[turn.side]
        confidence_bar = self._confidence_bar(turn.confidence)

        # Build header
        header = (
            f"[bold]Turn {turn.turn_number}[/bold]  |  "
            f"[bold {color}]{turn.agent_name.upper()}[/bold {color}]  |  "
            f"{turn.agent_role}  |  [{color}]{label}[/{color}]"
        )

        # Thinking block (dim, italic)
        thinking_section = ""
        if turn.thinking:
            thinking_section = (
                f"[dim italic]💭 THINKING (internal)\n{turn.thinking}[/dim italic]\n\n"
            )

        # Rebuttal info
        rebuttal_section = ""
        if turn.rebuttal_targets:
            rebutted_claims = []
            for claim_id in turn.rebuttal_targets:
                claim = next((c for c in state.claim_ledger if c.claim_id == claim_id), None)
                if claim:
                    rebutted_claims.append(f"{claim_id} ({claim.agent_name})")
                else:
                    rebutted_claims.append(claim_id)
            rebuttal_section = f"\n\n[dim]🎯 REBUTTING: {', '.join(rebutted_claims)}[/dim]"

        # Confidence bar
        conf_section = f"\n[dim]📊 Confidence: {confidence_bar} {turn.confidence:.2f}[/dim]"

        # Citations
        citation_section = ""
        if turn.citations:
            lines = []
            for i, cit in enumerate(turn.citations, 1):
                lines.append(f"  [{i}] [link={cit.url}]{cit.title}[/link]  [dim]{cit.url}[/dim]")
                if cit.snippet:
                    lines.append(f"      [dim italic]{cit.snippet[:120]}[/dim italic]")
            citation_section = "\n\n[bold dim]🔗 SOURCES[/bold dim]\n" + "\n".join(lines)

        body = (
            f"{thinking_section}"
            f"[bold]🗣 ARGUMENT[/bold]\n{turn.argument}"
            f"{rebuttal_section}"
            f"{conf_section}"
            f"{citation_section}"
        )

        console.print()
        console.print(
            Panel(
                body,
                title=header,
                border_style=color,
                padding=(1, 2),
            )
        )

        # Show updated claim ledger after each turn
        self._show_claim_ledger(state.claim_ledger)

    # ─────────────────────────────────────────────
    # Mediator Turn
    # ─────────────────────────────────────────────

    def show_mediator_turn(self, turn: DebateTurn):
        header = (
            f"[bold]Turn {turn.turn_number}[/bold]  |  "
            f"[bold yellow]{turn.agent_name.upper()}[/bold yellow]  |  [yellow]MEDIATOR[/yellow]"
        )

        thinking_section = ""
        if turn.thinking:
            thinking_section = (
                f"[dim italic]💭 THINKING (internal)\n{turn.thinking}[/dim italic]\n\n"
            )

        body = f"{thinking_section}[bold]⚖️  MID-DEBATE ANALYSIS[/bold]\n{turn.argument}"

        console.print()
        console.print(
            Panel(
                body,
                title=header,
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # ─────────────────────────────────────────────
    # Final Verdict
    # ─────────────────────────────────────────────

    def show_final_verdict(self, verdict: FinalVerdict, state: DebateState):
        console.print()
        console.print(Rule("[bold yellow]FINAL VERDICT[/bold yellow]", style="yellow"))
        console.print()

        # Argument scorecard
        if verdict.argument_scorecard:
            score_table = Table(
                title="Argument Scorecard",
                box=box.ROUNDED,
                border_style="yellow",
                show_header=True,
                header_style="bold yellow",
            )
            score_table.add_column("Claim ID", style="dim")
            score_table.add_column("Summary")
            score_table.add_column("Rating", justify="center")
            score_table.add_column("Rebutted?", justify="center")
            score_table.add_column("Notes")

            for score in verdict.argument_scorecard:
                rating_str = "★" * score.defense_rating + "☆" * (5 - score.defense_rating)
                rebutted_str = (
                    "[red]Yes[/red]" if score.successfully_rebutted else "[green]No[/green]"
                )
                score_table.add_row(
                    score.claim_id,
                    score.claim_summary[:60] + ("..." if len(score.claim_summary) > 60 else ""),
                    rating_str,
                    rebutted_str,
                    score.notes[:80] + ("..." if len(score.notes) > 80 else ""),
                )
            console.print(score_table)
            console.print()

        # Strongest arguments
        if verdict.strongest_pro_arguments:
            pro_args = "\n".join(f"  • {a}" for a in verdict.strongest_pro_arguments)
            console.print(
                Panel(
                    pro_args, title="[green]Strongest PRO Arguments[/green]", border_style="green"
                )
            )

        if verdict.strongest_con_arguments:
            con_args = "\n".join(f"  • {a}" for a in verdict.strongest_con_arguments)
            console.print(
                Panel(con_args, title="[red]Strongest CON Arguments[/red]", border_style="red")
            )

        # Fatal weaknesses
        console.print(
            Panel(
                f"[green]PRO:[/green] {verdict.fatal_pro_weaknesses}\n\n"
                f"[red]CON:[/red] {verdict.fatal_con_weaknesses}",
                title="[bold]Fatal Weaknesses[/bold]",
                border_style="dim",
            )
        )

        # Confidence trajectory
        console.print(
            Panel(
                verdict.confidence_trajectory,
                title="[bold]Confidence Trajectory[/bold]",
                border_style="dim",
            )
        )

        # Nuanced conclusion
        console.print(
            Panel(
                verdict.nuanced_conclusion,
                title="[bold yellow]Nuanced Conclusion[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # Final verdict
        console.print(
            Panel(
                f"[bold]{verdict.verdict}[/bold]",
                title="[bold yellow]VERDICT[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # Verdict citations
        if verdict.citations:
            lines = []
            for i, cit in enumerate(verdict.citations, 1):
                lines.append(f"  [{i}] [link={cit.url}]{cit.title}[/link]  [dim]{cit.url}[/dim]")
                if cit.snippet:
                    lines.append(f"      [dim italic]{cit.snippet[:120]}[/dim italic]")
            console.print(
                Panel(
                    "\n".join(lines),
                    title="[bold dim]🔗 VERDICT SOURCES[/bold dim]",
                    border_style="dim",
                )
            )

    # ─────────────────────────────────────────────
    # Claim Ledger
    # ─────────────────────────────────────────────

    def _show_claim_ledger(self, claims: list[Claim]):
        if not claims:
            return

        table = Table(
            title="Claim Ledger",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
            expand=False,
        )
        table.add_column("ID", style="dim", width=10)
        table.add_column("Agent", width=20)
        table.add_column("Side", width=6)
        table.add_column("Status", width=10)
        table.add_column("Claim", no_wrap=False)

        for claim in claims:
            color = SIDE_COLORS.get(claim.side, "white")
            if claim.rebutted_by:
                status = "[dim]REBUTTED[/dim]"
                claim_text = f"[dim strike]{claim.content[:80]}[/dim strike]"
            else:
                status = f"[{color}]STANDING[/{color}]"
                claim_text = claim.content[:80] + ("..." if len(claim.content) > 80 else "")

            table.add_row(
                claim.claim_id,
                claim.agent_name,
                f"[{color}]{claim.side.upper()}[/{color}]",
                status,
                claim_text,
            )

        console.print()
        console.print(Padding(table, (0, 0, 0, 2)))

    # ─────────────────────────────────────────────
    # Misc
    # ─────────────────────────────────────────────

    def show_interrupt_notice(self):
        console.print()
        console.print("[yellow]Debate interrupted. Saving partial transcript...[/yellow]")

    def show_transcript_saved(self, path: str):
        console.print()
        console.print(f"[green]Transcript saved to:[/green] {path}")

    def show_error(self, message: str):
        console.print(f"[bold red]ERROR:[/bold red] {message}")

    def _confidence_bar(self, confidence: float, width: int = 10) -> str:
        filled = int(confidence * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar
