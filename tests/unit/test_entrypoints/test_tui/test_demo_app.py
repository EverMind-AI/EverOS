"""EverOS demo TUI color contracts."""

from __future__ import annotations

import pytest

from everos.entrypoints.tui.demo.app import (
    SPHERE_FRAME_HEIGHT,
    SPHERE_FRAME_WIDTH,
    TERMINAL_CELL_HEIGHT_RATIO,
    DotSphereWidget,
    EverOSDemoApp,
    _field_header_text,
    _hero_text,
    _payoff_text,
    _recall_proof_text,
    _signal_rail_text,
    _source_tree_text,
    _sphere_caption,
)
from everos.entrypoints.tui.demo.data import build_demo_story
from everos.entrypoints.tui.demo.widgets.sphere import SPHERE_STATES


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
    visual_ratio = (SPHERE_FRAME_WIDTH - 4) / (
        SPHERE_FRAME_HEIGHT * TERMINAL_CELL_HEIGHT_RATIO
    )

    assert visual_ratio == pytest.approx(1.0, abs=0.04)
    assert SPHERE_FRAME_WIDTH == 37
    assert SPHERE_FRAME_HEIGHT == 17


def test_demo_tui_celebrates_after_source_reveal() -> None:
    assert DotSphereWidget.STATES[-2:] == ("source", "celebrating")
    assert set(DotSphereWidget.STATES).issubset(SPHERE_STATES)


def test_demo_tui_renders_playable_story_copy() -> None:
    story = build_demo_story(
        "I keep my Monday design review notes in Notion.",
        "Where are my Monday review notes?",
    )

    assert "user=you" in _field_header_text(story).plain
    assert "Where are my Monday review notes?" in _sphere_caption(story).plain
    assert story.answer in _sphere_caption(story).plain
    assert "server wake" not in _signal_rail_text(story).plain
    assert "memory core" in _signal_rail_text(story).plain
    assert story.source_filename in _source_tree_text(story).plain
    assert story.fact_filename in _source_tree_text(story).plain
    assert story.answer in _recall_proof_text(story).plain
    assert story.answer in _payoff_text(story).plain


def test_demo_tui_signal_rail_keeps_source_status_columns_separate() -> None:
    rail = _signal_rail_text().plain

    assert "mdattached" not in rail
    assert "md7 nodes" not in rail
    assert "..." in rail


def _css_block(css: str, selector: str) -> str:
    start = css.index(f"{selector} {{")
    end = css.index("}", start)
    return css[start:end]
