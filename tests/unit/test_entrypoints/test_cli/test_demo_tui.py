"""EverOS demo TUI color contracts."""

from __future__ import annotations

import pytest

from everos.entrypoints.cli.demo_tui import (
    SPHERE_FRAME_HEIGHT,
    SPHERE_FRAME_WIDTH,
    TERMINAL_CELL_HEIGHT_RATIO,
    DotSphereWidget,
    EverOSDemoApp,
    _hero_text,
    _payoff_text,
    _signal_rail_text,
)


def test_demo_tui_uses_poster_derived_brand_palette() -> None:
    css = EverOSDemoApp.CSS

    assert "#F9B91C" in css
    assert "#31302B" in css
    assert "#F5EDDC" in css
    assert "#918C80" in css
    assert "#FFE600" not in css
    assert "#55D6FF" not in css
    assert "#73F7A7" not in css
    assert any(span.style == "bold #F9B91C" for span in _hero_text().spans)


def test_demo_tui_uses_elevated_instrument_layout() -> None:
    css = EverOSDemoApp.CSS

    assert "#command-strip" in css
    assert "#memory-field" in css
    assert "#signal-rail" in css
    assert "#provenance-strip" in css
    assert "#payoff" in css
    assert "FooterKey" in css
    assert "background: #F9B91C" in css
    assert any("on #F9B91C" in span.style for span in _hero_text().spans)
    assert "border: double" not in css
    assert "border: heavy" not in css


def test_demo_tui_uses_balanced_panel_proportions() -> None:
    css = EverOSDemoApp.CSS

    command_strip = _css_block(css, "#command-strip")
    signal_rail = _css_block(css, "#signal-rail")
    payoff = _css_block(css, "#payoff")

    assert "height: 2;" in command_strip
    assert "border-left: thick" not in command_strip
    assert "background: #31302B" not in command_strip
    assert len(_hero_text().plain.splitlines()) == 1
    assert len(_hero_text().plain) <= 56

    assert "height: 1fr;" in DotSphereWidget.DEFAULT_CSS

    assert "height: 100%;" in signal_rail
    assert "source route" in _signal_rail_text().plain
    assert "recall proof" in _signal_rail_text().plain

    assert "height: 2;" in payoff
    assert "background: #24231E;" in payoff
    assert "padding: 0 1;" in payoff
    assert _payoff_text().plain.startswith("memory formed:")
    assert "bold #F9B91C" in {span.style for span in _payoff_text().spans}


def test_demo_tui_sphere_renders_round_in_terminal_cells() -> None:
    visual_ratio = (SPHERE_FRAME_WIDTH - 2) / (
        SPHERE_FRAME_HEIGHT * TERMINAL_CELL_HEIGHT_RATIO
    )

    assert visual_ratio == pytest.approx(1.0, abs=0.04)
    assert SPHERE_FRAME_WIDTH == 37
    assert SPHERE_FRAME_HEIGHT == 17


def _css_block(css: str, selector: str) -> str:
    start = css.index(f"{selector} {{")
    end = css.index("}", start)
    return css[start:end]
