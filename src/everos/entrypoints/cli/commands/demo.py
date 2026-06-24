"""``everos demo`` — first-run memory sphere demo.

The default command launches an interactive Textual TUI: the user types memories
and recall questions directly in the UI, and each round runs the *real* memory
pipeline against a hosted EverOS server (keys live server-side; see
:mod:`everos.entrypoints.tui.demo.cloud`). ``--plain`` / ``--cinematic`` are
static, no-network renderings for non-interactive shells and README media.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel

from everos.entrypoints.tui.demo import cloud
from everos.entrypoints.tui.demo.data import DemoStory, default_demo_story
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
            help="Launch the looping README-style showcase (no input box).",
        ),
        live: bool = typer.Option(
            False,
            "--live",
            help="Run the interactive flow against your own EverOS server.",
        ),
        cloud_mode: bool = typer.Option(
            False,
            "--cloud",
            help="Run against EverMind's hosted demo server (this is the default).",
        ),
        server_url: str = typer.Option(
            cloud.LIVE_DEMO_SERVER_URL,
            "--server-url",
            help="EverOS server URL used by --live (and to override --cloud).",
        ),
    ) -> None:
        """Launch the EverOS first-memory Textual TUI."""
        if plain or not sys.stdout.isatty():
            _print_plain_demo()
            return

        if cinematic:
            _load_run_demo_tui()()
            return

        _launch_interactive_demo(live=live, server_url=server_url)


def _launch_interactive_demo(*, live: bool, server_url: str) -> None:
    """Launch the cloud-backed interactive TUI, or --live against your own server."""

    run_demo_tui = _load_run_demo_tui()
    if live:
        base_url = server_url
        session_id, user_id = cloud.LIVE_DEMO_SESSION_ID, cloud.LIVE_DEMO_USER_ID
    else:
        base_url = cloud.resolve_cloud_base_url(server_url)
        session_id, user_id = cloud.new_demo_identity()

    run_demo_tui(
        interactive=True,
        base_url=base_url,
        session_id=session_id,
        user_id=user_id,
    )


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
