"""``everos demo`` — first-run memory sphere demo.

The default command launches an interactive Textual TUI: the user types memories
and recall questions directly in the UI, and each round runs the *real* memory
pipeline against a hosted EverOS server (keys live server-side; see
:mod:`everos.entrypoints.tui.demo.cloud`). ``--plain`` / ``--cinematic`` are
static, no-network renderings for non-interactive shells and README media.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel

from everos.entrypoints.cli._log_setup import configure_cli_logging
from everos.entrypoints.tui.demo import cloud
from everos.entrypoints.tui.demo.data import DemoStory, default_demo_story
from everos.entrypoints.tui.demo.widgets.sphere import (
    EVEROS_GREEN,
    EVEROS_YELLOW,
    build_dot_sphere,
    render_dot_sphere_text,
)

TEXTUAL_DISABLE_KITTY_KEY_ENV = "TEXTUAL_DISABLE_KITTY_KEY"


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
            help="Use your own EverOS Cloud API key (env EVEROS_CLOUD_API_KEY).",
        ),
        cloud_mode: bool = typer.Option(
            False,
            "--cloud",
            help="Run against EverOS Cloud with the demo key (this is the default).",
        ),
        server_url: str = typer.Option(
            cloud.LIVE_DEMO_SERVER_URL,
            "--server-url",
            help="Override the EverOS Cloud API base URL.",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Emit INFO-level lifecycle logs (default: WARNING only).",
        ),
    ) -> None:
        """Launch the EverOS first-memory Textual TUI."""
        configure_cli_logging(verbose=verbose)
        if plain or not sys.stdout.isatty():
            _print_plain_demo()
            return

        user_label = _resolve_local_user()
        if cinematic:
            _load_run_demo_tui()(user_label=user_label)
            return

        _launch_interactive_demo(
            live=live, server_url=server_url, user_label=user_label
        )


def _launch_interactive_demo(
    *, live: bool, server_url: str, user_label: str = "you"
) -> None:
    """Launch the cloud-platform interactive TUI.

    The default mode talks to the credential-injecting public relay. ``--live``
    bypasses the relay and uses the user's own platform key directly.
    """

    run_demo_tui = _load_run_demo_tui()
    base_url = (
        cloud.resolve_live_base_url(server_url)
        if live
        else cloud.resolve_cloud_base_url(server_url)
    )
    session_id, user_id = cloud.new_demo_identity()
    api_key = cloud.resolve_user_key() if live else cloud.resolve_demo_key()

    run_demo_tui(
        interactive=True,
        base_url=base_url,
        session_id=session_id,
        user_id=user_id,
        api_key=api_key,
        user_label=user_label,
    )


def _resolve_local_user() -> str:
    """Local-first display name: the clone's git identity, else the OS user."""

    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        name = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    if name:
        return name
    try:
        return getpass.getuser()
    except Exception:
        return "you"


def _load_run_demo_tui():
    # Textual's Kitty extended-key parser conflicts with macOS Chinese IMEs in
    # some terminals: the pinyin pre-edit is delivered as ordinary key presses
    # before the selected Han characters are committed. Configure Textual
    # before its first import so `everos demo` accepts composed Chinese input.
    # Keep an explicit user override intact for terminals that need the
    # extended-key protocol.
    os.environ.setdefault(TEXTUAL_DISABLE_KITTY_KEY_ENV, "1")
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
