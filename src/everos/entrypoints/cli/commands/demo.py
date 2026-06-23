"""``everos demo`` — first-run memory sphere demo."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel

from everos.entrypoints.tui.demo.data import (
    DEFAULT_MEMORY_SEED,
    DEFAULT_QUERY,
    DemoStory,
    build_demo_story,
    default_demo_story,
)
from everos.entrypoints.tui.demo.widgets.sphere import (
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
        cinematic: bool = typer.Option(
            False,
            "--cinematic",
            help="Skip prompts and launch the looping README-style demo.",
        ),
    ) -> None:
        """Launch the EverOS first-memory Textual TUI."""
        if plain or not sys.stdout.isatty():
            _print_plain_demo()
            return

        _run_interactive_demo(cinematic=cinematic)


def _run_interactive_demo(*, cinematic: bool) -> None:
    run_demo_tui = _load_run_demo_tui()
    story = None if cinematic else _collect_playable_story()
    run_demo_tui(story=story)


def _load_run_demo_tui():
    try:
        from everos.entrypoints.tui.demo.app import run_demo_tui
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

    return run_demo_tui


def _collect_playable_story() -> DemoStory:
    Console().print(
        f"[bold {EVEROS_YELLOW}]EverOS demo[/] "
        "Give it one memory, then ask for it back."
    )
    memory = typer.prompt(
        "Give EverOS one thing to remember",
        default=DEFAULT_MEMORY_SEED,
    )
    query = typer.prompt(
        "Ask EverOS to recall it",
        default=DEFAULT_QUERY,
    )
    return build_demo_story(memory, query)


def _print_plain_demo(story: DemoStory | None = None) -> None:
    story = story or default_demo_story()
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
    console.print(story.memory)
    console.print()
    console.print(f"[bold {EVEROS_YELLOW}]Source:[/] {story.source_filename}")
