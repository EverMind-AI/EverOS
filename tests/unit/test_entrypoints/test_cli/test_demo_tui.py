"""EverOS demo TUI color contracts."""

from __future__ import annotations

import pytest

from everos.entrypoints.cli.demo_tui import (
    SPHERE_FRAME_HEIGHT,
    SPHERE_FRAME_WIDTH,
    TERMINAL_CELL_HEIGHT_RATIO,
    EverOSDemoApp,
    _hero_text,
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
    assert "border-left: thick #F9B91C" in css
    assert "FooterKey" in css
    assert "background: #F9B91C" in css
    assert "border: double" not in css
    assert "border: heavy" not in css


def test_demo_tui_sphere_renders_round_in_terminal_cells() -> None:
    visual_ratio = SPHERE_FRAME_WIDTH / (
        SPHERE_FRAME_HEIGHT * TERMINAL_CELL_HEIGHT_RATIO
    )

    assert visual_ratio == pytest.approx(1.0, abs=0.04)
