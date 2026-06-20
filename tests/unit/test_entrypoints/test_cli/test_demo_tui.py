"""EverOS demo TUI color contracts."""

from __future__ import annotations

from everos.entrypoints.cli.demo_tui import EverOSDemoApp, _hero_text


def test_demo_tui_uses_bright_yellow_brand_primary() -> None:
    assert "#FFE600" in EverOSDemoApp.CSS
    assert "#FFD23F" not in EverOSDemoApp.CSS
    assert any(span.style == "bold #FFE600" for span in _hero_text().spans)


def test_demo_tui_uses_elevated_instrument_layout() -> None:
    css = EverOSDemoApp.CSS

    assert "#command-strip" in css
    assert "#memory-field" in css
    assert "#signal-rail" in css
    assert "#provenance-strip" in css
    assert "#payoff" in css
    assert "border-left: thick #FFE600" in css
    assert "FooterKey" in css
    assert "background: #FFE600" in css
    assert "border: double" not in css
    assert "border: heavy" not in css
