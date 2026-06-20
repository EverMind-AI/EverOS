"""``everos demo`` — first-run memory sphere demo."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel

from everos.entrypoints.cli.demo_sphere import (
    EVEROS_GREEN,
    EVEROS_YELLOW,
    build_dot_sphere,
    render_dot_sphere_text,
)


def register(parent: typer.Typer) -> None:
    """Attach the ``demo`` command to the root CLI app."""

    @parent.command("demo")
    def demo(
        plain: bool = typer.Option(
            False,
            "--plain",
            help="Print a static terminal preview instead of launching the TUI.",
        ),
    ) -> None:
        """Launch the EverOS first-memory Textual TUI."""
        if plain or not sys.stdout.isatty():
            _print_plain_demo()
            return

        try:
            from everos.entrypoints.cli.demo_tui import run_demo_tui
        except ModuleNotFoundError as exc:
            if exc.name != "textual":
                raise
            typer.secho(
                "error: Textual is required for `everos demo`; install the "
                "package with TUI dependencies or run `everos demo --plain`.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from exc

        run_demo_tui()


def _print_plain_demo() -> None:
    console = Console()
    frame = build_dot_sphere(
        width=57,
        height=23,
        phase=0.18,
        state_key="remembered",
    )
    console.print(
        Panel(
            render_dot_sphere_text(frame),
            title="EverOS Memory Sphere",
            border_style=EVEROS_YELLOW,
        )
    )
    console.print(f"[bold {EVEROS_GREEN}]EverOS remembered:[/]")
    console.print("Alice likes climbing in Yosemite every spring.")
    console.print()
    console.print(f"[bold {EVEROS_YELLOW}]Source:[/] episode-2026-06-20.md")
