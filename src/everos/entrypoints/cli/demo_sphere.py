"""Dot-sphere primitives for the EverOS demo TUI.

The Textual app consumes these pure rendering primitives so the animated
surface stays testable without standing up a terminal UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rich.text import Text

EVEROS_YELLOW = "#F9B91C"
EVEROS_YELLOW_SOFT = "#F6C23B"
EVEROS_AMBER_DIM = "#4A3D20"
EVEROS_AMBER = "#8B763F"
EVEROS_CYAN = "#F5EDDC"
EVEROS_GREEN = "#D8CDAF"
EVEROS_ORANGE = "#C09525"


@dataclass(frozen=True)
class SphereState:
    """Visual and copy settings for a sphere animation state."""

    key: str
    caption: str
    accent: str


@dataclass(frozen=True)
class DotCell:
    """One projected dot in terminal cell coordinates."""

    x: int
    y: int
    z: float
    glyph: str
    style: str
    highlighted: bool = False


@dataclass(frozen=True)
class DotSphereFrame:
    """A fully projected dot-sphere frame."""

    width: int
    height: int
    state: SphereState
    cells: tuple[DotCell, ...]

    @property
    def caption(self) -> str:
        return self.state.caption


SPHERE_STATES: dict[str, SphereState] = {
    "booting": SphereState(
        key="booting",
        caption="forming local memory field",
        accent=EVEROS_YELLOW,
    ),
    "ingesting": SphereState(
        key="ingesting",
        caption="ingesting conversation dots",
        accent=EVEROS_CYAN,
    ),
    "extracting": SphereState(
        key="extracting",
        caption="extracting episode -> atomic facts",
        accent=EVEROS_ORANGE,
    ),
    "indexing": SphereState(
        key="indexing",
        caption="syncing SQLite + LanceDB orbit",
        accent=EVEROS_CYAN,
    ),
    "recalling": SphereState(
        key="recalling",
        caption="scanning memory sphere",
        accent=EVEROS_GREEN,
    ),
    "remembered": SphereState(
        key="remembered",
        caption="remembered Yosemite preference",
        accent=EVEROS_YELLOW,
    ),
    "source": SphereState(
        key="source",
        caption="revealing episode.md source",
        accent=EVEROS_YELLOW_SOFT,
    ),
}


def build_dot_sphere(
    *, width: int, height: int, phase: float, state_key: str
) -> DotSphereFrame:
    """Build one dot-sphere animation frame."""
    if width < 13 or height < 7:
        raise ValueError("dot sphere requires at least 13x7 cells")
    try:
        state = SPHERE_STATES[state_key]
    except KeyError as exc:
        raise ValueError(f"unknown sphere state: {state_key}") from exc

    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    radius_x = center_x
    radius_y = center_y
    rotation = phase * math.tau
    active_target = _highlight_target(width, height)

    by_position: dict[tuple[int, int], DotCell] = {}
    for y in range(height):
        yn = (y - center_y) / radius_y
        if abs(yn) > 1:
            continue
        ring_radius = math.sqrt(max(0.0, 1.0 - yn * yn))
        dot_count = max(4, round(ring_radius * width * 0.72))
        for index in range(dot_count):
            theta = math.tau * (index / dot_count) + rotation + y * 0.37
            x3 = ring_radius * math.cos(theta)
            z3 = ring_radius * math.sin(theta)
            x = round(center_x + x3 * radius_x)
            style = _style_for_depth(z3, state)
            glyph = _glyph_for_depth(z3)
            highlighted = False

            if (
                state.key in {"recalling", "remembered", "source"}
                and (
                    x,
                    y,
                )
                == active_target
            ):
                highlighted = True
                glyph = "◆"
                style = (
                    f"bold {EVEROS_CYAN}"
                    if state.key == "recalling"
                    else f"bold {EVEROS_YELLOW}"
                )

            existing = by_position.get((x, y))
            if existing is None or z3 > existing.z or highlighted:
                by_position[(x, y)] = DotCell(
                    x=x,
                    y=y,
                    z=z3,
                    glyph=glyph,
                    style=style,
                    highlighted=highlighted,
                )

    if state.key in {"recalling", "remembered", "source"} and not any(
        cell.highlighted for cell in by_position.values()
    ):
        hx, hy = active_target
        by_position[(hx, hy)] = DotCell(
            x=hx,
            y=hy,
            z=1.0,
            glyph="◆",
            style=(
                f"bold {EVEROS_CYAN}"
                if state.key == "recalling"
                else f"bold {EVEROS_YELLOW}"
            ),
            highlighted=True,
        )

    return DotSphereFrame(
        width=width,
        height=height,
        state=state,
        cells=tuple(sorted(by_position.values(), key=lambda cell: (cell.y, cell.x))),
    )


def render_dot_sphere_lines(frame: DotSphereFrame) -> list[list[DotCell | None]]:
    """Render cells into a sparse row grid for Rich/Textual consumers."""
    grid: list[list[DotCell | None]] = [
        [None for _ in range(frame.width)] for _ in range(frame.height)
    ]
    for cell in frame.cells:
        if 0 <= cell.x < frame.width and 0 <= cell.y < frame.height:
            grid[cell.y][cell.x] = cell
    return grid


def render_dot_sphere_text(frame: DotSphereFrame) -> Text:
    """Convert a frame into styled terminal text."""
    rows = render_dot_sphere_lines(frame)
    text = Text(no_wrap=True)
    for row in rows:
        for cell in row:
            if cell is None:
                text.append(" ")
            else:
                text.append(cell.glyph, style=cell.style)
        text.append("\n")
    text.append("\n")
    text.append(frame.caption, style=f"bold {frame.state.accent}")
    return text


def _style_for_depth(z: float, state: SphereState) -> str:
    if state.key == "extracting" and z > 0.38:
        return f"bold {EVEROS_ORANGE}"
    if state.key == "indexing" and z > 0.45:
        return f"bold {EVEROS_CYAN}"
    if state.key == "ingesting" and z > 0.5:
        return f"bold {EVEROS_CYAN}"
    if z > 0.58:
        return f"bold {EVEROS_YELLOW}"
    if z > 0.05:
        return f"{EVEROS_YELLOW}"
    if z > -0.45:
        return f"{EVEROS_AMBER}"
    return f"{EVEROS_AMBER_DIM}"


def _glyph_for_depth(z: float) -> str:
    if z > 0.62:
        return "●"
    if z > 0.1:
        return "•"
    return "·"


def _highlight_target(width: int, height: int) -> tuple[int, int]:
    return (round((width - 1) * 0.66), round((height - 1) * 0.42))
