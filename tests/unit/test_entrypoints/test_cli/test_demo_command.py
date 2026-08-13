"""EverOS demo command contracts."""

from __future__ import annotations

import os
import re

import typer
from rich.panel import Panel
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import demo as demo_command
from everos.entrypoints.tui.demo import cloud
from everos.entrypoints.tui.demo.data import DemoStory


def test_demo_help_exposes_all_modes() -> None:
    app = typer.Typer()
    demo_command.register(app)

    result = CliRunner().invoke(app, ["demo", "--help"], terminal_width=120)

    help_text = _strip_ansi(result.stdout)
    assert result.exit_code == 0
    for flag in ("--cinematic", "--live", "--cloud", "--server-url", "--verbose"):
        assert flag in help_text


def test_demo_configures_requested_log_level(monkeypatch) -> None:
    configured: list[bool] = []
    monkeypatch.setattr(
        demo_command,
        "configure_cli_logging",
        lambda *, verbose: configured.append(verbose),
    )
    monkeypatch.setattr(demo_command, "_print_plain_demo", lambda: None)
    app = typer.Typer()
    demo_command.register(app)

    result = CliRunner().invoke(app, ["--plain", "--verbose"])

    assert result.exit_code == 0
    assert configured == [True]


def test_plain_demo_uses_poster_gold_brand_primary(monkeypatch) -> None:
    printed: list[object] = []

    class FakeConsole:
        def print(self, *renderables: object, **_: object) -> None:
            printed.extend(renderables)

    monkeypatch.setattr(demo_command, "Console", FakeConsole)

    demo_command._print_plain_demo()

    panel = next(item for item in printed if isinstance(item, Panel))
    printed_text = "\n".join(str(item) for item in printed)
    assert panel.border_style == "#F9B91C"
    assert "#F9B91C" in printed_text
    assert "#FFE600" not in printed_text


def test_plain_demo_prints_custom_story(monkeypatch) -> None:
    printed: list[object] = []

    class FakeConsole:
        def print(self, *renderables: object, **_: object) -> None:
            printed.extend(renderables)

    monkeypatch.setattr(demo_command, "Console", FakeConsole)

    demo_command._print_plain_demo(
        DemoStory(
            owner="you",
            memory="I keep my Monday design review notes in Notion.",
            query="Where are my Monday review notes?",
            answer="In Notion.",
            source_filename="episode-demo.md",
            fact_filename="atomic_fact-demo.md",
        )
    )

    printed_text = "\n".join(str(item) for item in printed)
    assert "EverOS remembered" in printed_text
    assert "I keep my Monday design review notes in Notion." in printed_text
    assert "episode-demo.md" in printed_text


def test_launch_interactive_defaults_to_cloud_with_unique_identity(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        demo_command,
        "_load_run_demo_tui",
        lambda: lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.delenv(cloud.CLOUD_DEMO_SERVER_URL_ENV, raising=False)
    monkeypatch.setenv(cloud.CLOUD_DEMO_KEY_ENV, "demo-key")

    demo_command._launch_interactive_demo(
        live=False, server_url=cloud.LIVE_DEMO_SERVER_URL
    )

    assert captured["interactive"] is True
    assert captured["base_url"] == cloud.CLOUD_API_BASE_URL
    assert str(captured["session_id"]).startswith("everos-demo-")
    assert str(captured["user_id"]).startswith("everos_demo_")
    assert captured["api_key"] == "demo-key"  # optional direct-test override


def test_launch_interactive_live_uses_own_cloud_key(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        demo_command,
        "_load_run_demo_tui",
        lambda: lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.delenv(cloud.CLOUD_DEMO_SERVER_URL_ENV, raising=False)
    monkeypatch.setenv(cloud.CLOUD_USER_KEY_ENV, "user-key")

    demo_command._launch_interactive_demo(
        live=True, server_url=cloud.LIVE_DEMO_SERVER_URL
    )

    # --live bypasses the public relay and hits the platform with the user's key.
    assert captured["base_url"] == cloud.CLOUD_PLATFORM_API_BASE_URL
    assert captured["api_key"] == "user-key"
    assert str(captured["session_id"]).startswith("everos-demo-")


def test_loading_demo_tui_disables_kitty_keys_for_ime_compatibility(
    monkeypatch,
) -> None:
    monkeypatch.delenv(demo_command.TEXTUAL_DISABLE_KITTY_KEY_ENV, raising=False)

    demo_command._load_run_demo_tui()

    assert os.environ[demo_command.TEXTUAL_DISABLE_KITTY_KEY_ENV] == "1"


def test_loading_demo_tui_preserves_explicit_kitty_key_override(monkeypatch) -> None:
    monkeypatch.setenv(demo_command.TEXTUAL_DISABLE_KITTY_KEY_ENV, "0")

    demo_command._load_run_demo_tui()

    assert os.environ[demo_command.TEXTUAL_DISABLE_KITTY_KEY_ENV] == "0"


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
