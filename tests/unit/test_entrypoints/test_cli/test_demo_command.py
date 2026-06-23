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


def test_demo_help_exposes_live_mode() -> None:
    app = typer.Typer()
    demo_command.register(app)

    result = CliRunner().invoke(app, ["demo", "--help"], terminal_width=120)

    help_text = _strip_ansi(result.stdout)
    assert result.exit_code == 0
    assert "--live" in help_text
    assert "--server-url" in help_text


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


def test_live_demo_flow_calls_server_and_builds_story() -> None:
    story = build_demo_story(
        "I love climbing in Yosemite every spring.",
        "Where do I like to climb?",
    )
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        base_url: str,
        json_body: dict[str, object] | None = None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        calls.append((method, path, json_body))
        assert base_url == "http://server.test"
        assert timeout_seconds == 3.0
        if path == "/health":
            return {"status": "ok"}
        if path == "/api/v1/memory/add":
            return {"message_count": 1, "status": "accumulated"}
        if path == "/api/v1/memory/flush":
            return {"message_count": 1, "status": "extracted"}
        if path == "/api/v1/memory/search":
            return {
                "data": {
                    "episodes": [
                        {
                            "id": "alice_ep_20260623_0001",
                            "episode": "Alice loves climbing in Yosemite every spring.",
                            "summary": "Alice climbs in Yosemite every spring.",
                            "subject": "Yosemite climbing",
                            "score": 0.82,
                            "atomic_facts": [
                                {
                                    "id": "alice_af_20260623_0001",
                                    "content": (
                                        "Alice loves climbing in Yosemite every spring."
                                    ),
                                    "score": 0.91,
                                }
                            ],
                        }
                    ]
                }
            }
        raise AssertionError(f"unexpected request: {method} {path}")

    live_story = demo_command._run_live_demo_flow(
        story,
        base_url="http://server.test",
        request_json=fake_request,
        timeout_seconds=3.0,
    )

    assert [path for _, path, _ in calls] == [
        "/health",
        "/api/v1/memory/add",
        "/api/v1/memory/flush",
        "/api/v1/memory/search",
    ]
    add_body = calls[1][2]
    assert add_body is not None
    assert add_body["session_id"] == "everos-demo-live"
    assert live_story.answer == "Alice loves climbing in Yosemite every spring."
    assert live_story.source_filename == "episode:alice_ep_20260623_0001"
    assert live_story.fact_filename == "fact:alice_af_20260623_0001"


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
