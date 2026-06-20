"""EverOS demo command contracts."""

from __future__ import annotations

from rich.panel import Panel

from everos.entrypoints.cli.commands import demo as demo_command


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
