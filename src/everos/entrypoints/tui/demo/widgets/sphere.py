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
BRAILLE_BASE = 0x2800
BRAILLE_DOT_BITS = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)
SPHERE_POINT_COUNT = 1300
CONFETTI_POINT_COUNT = 150
CONFETTI_GLYPHS = (".", "+", "*", "x")
CONFETTI_STYLES = (
    EVEROS_YELLOW,
    EVEROS_YELLOW_SOFT,
    EVEROS_CYAN,
    EVEROS_ORANGE,
    EVEROS_AMBER,
)
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))


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
    "celebrating": SphereState(
        key="celebrating",
        caption="memory crystallized",
        accent=EVEROS_YELLOW,
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

    if state.key == "celebrating":
        return _build_confetti_burst(
            width=width,
            height=height,
            phase=phase,
            state=state,
        )

    sub_width = width * 2
    sub_height = height * 4
    sub_center_x = (sub_width - 1) / 2
    sub_center_y = (sub_height - 1) / 2
    radius_x = max(1.0, sub_center_x - 5)
    radius_y = max(1.0, sub_center_y - 3)
    rotation = phase * math.tau
    active_target = _highlight_target(width, height)

    masks: dict[tuple[int, int], int] = {}
    depths: dict[tuple[int, int], float] = {}
    highlighted_positions: set[tuple[int, int]] = set()
    for index in range(SPHERE_POINT_COUNT):
        y3 = 1 - 2 * ((index + 0.5) / SPHERE_POINT_COUNT)
        ring_radius = math.sqrt(max(0.0, 1.0 - y3 * y3))
        theta = index * GOLDEN_ANGLE + rotation
        x3 = ring_radius * math.cos(theta)
        z3 = ring_radius * math.sin(theta)
        sub_x = round(sub_center_x + x3 * radius_x)
        sub_y = round(sub_center_y + y3 * radius_y)
        if not (0 <= sub_x < sub_width and 0 <= sub_y < sub_height):
            continue
        _add_braille_dot(
            masks=masks,
            depths=depths,
            sub_x=sub_x,
            sub_y=sub_y,
            z=z3,
        )

    if state.key in {"recalling", "remembered", "source"}:
        highlighted_positions.add(active_target)
        target_sub_x = active_target[0] * 2 + 1
        target_sub_y = active_target[1] * 4 + 1
        _add_braille_dot(
            masks=masks,
            depths=depths,
            sub_x=target_sub_x,
            sub_y=target_sub_y,
            z=1.0,
        )

    cells = []
    for (x, y), mask in masks.items():
        highlighted = (x, y) in highlighted_positions
        if highlighted and state.key == "recalling":
            style = EVEROS_CYAN
        elif highlighted:
            style = EVEROS_YELLOW
        else:
            style = _style_for_depth(depths[(x, y)], state)
        cells.append(
            DotCell(
                x=x,
                y=y,
                z=depths[(x, y)],
                glyph=chr(BRAILLE_BASE + mask),
                style=style,
                highlighted=highlighted,
            )
        )

    return DotSphereFrame(
        width=width,
        height=height,
        state=state,
        cells=tuple(sorted(cells, key=lambda cell: (cell.y, cell.x))),
    )


def _build_confetti_burst(
    *, width: int, height: int, phase: float, state: SphereState
) -> DotSphereFrame:
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    radius_x = max(1.0, center_x - 3)
    radius_y = max(1.0, center_y - 2)
    local_phase = _state_local_phase(phase, state.key)
    bloom = 0.62 + 0.58 * math.sin(local_phase * math.pi)
    rotation = phase * math.tau * 1.4

    cells_by_position: dict[tuple[int, int], DotCell] = {}
    for index in range(CONFETTI_POINT_COUNT):
        shell = 0.55 + 0.45 * ((index % 17) / 16)
        angle = index * GOLDEN_ANGLE + rotation
        drift = math.sin(phase * math.tau * 2 + index * 0.23)
        x = round(center_x + math.cos(angle) * radius_x * shell * bloom)
        y = round(
            center_y
            + math.sin(angle) * radius_y * shell * bloom
            + drift * 0.75 * local_phase
        )
        if not (0 <= x < width and 0 <= y < height):
            continue

        z = math.cos(angle - rotation) * shell
        glyph = CONFETTI_GLYPHS[(index + int(local_phase * 10)) % len(CONFETTI_GLYPHS)]
        style = CONFETTI_STYLES[
            (index * 3 + int(local_phase * 7)) % len(CONFETTI_STYLES)
        ]
        position = (x, y)
        existing = cells_by_position.get(position)
        if existing is None or z > existing.z:
            cells_by_position[position] = DotCell(
                x=x,
                y=y,
                z=z,
                glyph=glyph,
                style=style,
            )

    return DotSphereFrame(
        width=width,
        height=height,
        state=state,
        cells=tuple(
            sorted(cells_by_position.values(), key=lambda cell: (cell.y, cell.x))
        ),
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


def _add_braille_dot(
    *,
    masks: dict[tuple[int, int], int],
    depths: dict[tuple[int, int], float],
    sub_x: int,
    sub_y: int,
    z: float,
) -> None:
    cell_x = sub_x // 2
    cell_y = sub_y // 4
    local_x = sub_x % 2
    local_y = sub_y % 4
    position = (cell_x, cell_y)
    masks[position] = masks.get(position, 0) | BRAILLE_DOT_BITS[local_x][local_y]
    depths[position] = max(z, depths.get(position, -1.0))


def _style_for_depth(z: float, state: SphereState) -> str:
    if state.key == "extracting" and z > 0.38:
        return EVEROS_ORANGE
    if state.key == "indexing" and z > 0.45:
        return EVEROS_CYAN
    if state.key == "ingesting" and z > 0.5:
        return EVEROS_CYAN
    if z > 0.58:
        return EVEROS_YELLOW
    if z > 0.05:
        return EVEROS_YELLOW
    return EVEROS_AMBER


def _highlight_target(width: int, height: int) -> tuple[int, int]:
    return (round((width - 1) * 0.66), round((height - 1) * 0.42))


def _state_local_phase(phase: float, state_key: str) -> float:
    state_keys = tuple(SPHERE_STATES)
    return (phase * len(state_keys) - state_keys.index(state_key)) % 1.0
