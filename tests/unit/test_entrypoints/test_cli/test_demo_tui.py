"""EverOS demo TUI color contracts."""

from __future__ import annotations

from everos.entrypoints.cli.demo_tui import EverOSDemoApp, _hero_text


def test_demo_tui_uses_bright_yellow_brand_primary() -> None:
    assert "#FFE600" in EverOSDemoApp.CSS
    assert "#FFD23F" not in EverOSDemoApp.CSS
    assert any(span.style == "bold #FFE600" for span in _hero_text().spans)
