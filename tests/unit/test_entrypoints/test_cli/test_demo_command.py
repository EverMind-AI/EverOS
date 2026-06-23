"""EverOS demo command contracts."""

from __future__ import annotations

import re

import pytest
import typer
from rich.panel import Panel
from typer.testing import CliRunner

from everos.entrypoints.cli.commands import demo as demo_command
from everos.entrypoints.tui.demo.data import build_demo_story


def test_demo_help_exposes_cinematic_mode() -> None:
    app = typer.Typer()
    demo_command.register(app)

    result = CliRunner().invoke(app, ["demo", "--help"], terminal_width=120)

    assert result.exit_code == 0
    assert "--cinematic" in _strip_ansi(result.stdout)


def test_collect_playable_story_prompts_for_memory_then_query(monkeypatch) -> None:
    prompts: list[tuple[str, str]] = []
    replies = iter(
        [
            "I keep my Monday design review notes in Notion.",
            "Where are my Monday review notes?",
        ]
    )

    def fake_prompt(label: str, *, default: str) -> str:
        prompts.append((label, default))
        return next(replies)

    monkeypatch.setattr(demo_command.typer, "prompt", fake_prompt)

    story = demo_command._collect_playable_story()

    assert [label for label, _ in prompts] == [
        "Give EverOS one thing to remember",
        "Ask EverOS to recall it",
    ]
    assert story.memory == "I keep my Monday design review notes in Notion."
    assert story.query == "Where are my Monday review notes?"


def test_interactive_demo_checks_textual_before_prompt(monkeypatch) -> None:
    def fail_load_tui() -> object:
        raise typer.Exit(code=1)

    def fail_prompt(*_: object, **__: object) -> str:
        pytest.fail("demo prompted before checking TUI availability")

    monkeypatch.setattr(demo_command, "_load_run_demo_tui", fail_load_tui)
    monkeypatch.setattr(demo_command.typer, "prompt", fail_prompt)

    with pytest.raises(typer.Exit):
        demo_command._run_interactive_demo(cinematic=False)


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
        build_demo_story(
            "I keep my Monday design review notes in Notion.",
            "Where are my Monday review notes?",
        )
    )

    printed_text = "\n".join(str(item) for item in printed)
    assert "EverOS remembered" in printed_text
    assert "I keep my Monday design review notes in Notion." in printed_text
    assert "episode-demo.md" in printed_text


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
