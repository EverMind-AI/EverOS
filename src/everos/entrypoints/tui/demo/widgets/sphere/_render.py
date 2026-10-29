"""Public frame API: build a state's frame, blend two frames, render to text.

``build_dot_sphere`` dispatches a state to its renderer; the blend eases
between states cell by cell instead of swapping the whole field.
"""

from __future__ import annotations

from rich.text import Text

from everos.entrypoints.tui.demo.widgets.sphere._models import (
    SPHERE_STATES,
    DotCell,
    DotSphereFrame,
    _state_local_phase,
)
from everos.entrypoints.tui.demo.widgets.sphere._styles import _blend_hex_color
from everos.entrypoints.tui.demo.widgets.sphere._supernova import _build_soft_supernova
from everos.entrypoints.tui.demo.widgets.sphere._working import _build_working_cloud


def build_dot_sphere(
    *,
    width: int,
    height: int,
    phase: float,
    state_key: str,
    state_phase: float | None = None,
) -> DotSphereFrame:
    """Build one dot-sphere animation frame."""
    if width < 13 or height < 7:
        raise ValueError("dot sphere requires at least 13x7 cells")
    try:
        state = SPHERE_STATES[state_key]
    except KeyError as exc:
        raise ValueError(f"unknown sphere state: {state_key}") from exc

    if state.key == "celebrating":
        return _build_soft_supernova(
            width=width,
            height=height,
            phase=phase,
            state=state,
            progress=(
                _state_local_phase(phase, state.key)
                if state_phase is None
                else state_phase
            ),
        )
    if state.key in {
        "booting",
        "ingesting",
        "extracting",
        "indexing",
        "recalling",
        "remembered",
        "source",
    }:
        return _build_working_cloud(
            width=width,
            height=height,
            phase=phase,
            state=state,
        )

    raise AssertionError(f"unhandled sphere state: {state.key}")


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


def blend_dot_sphere_frames(
    previous: DotSphereFrame,
    current: DotSphereFrame,
    progress: float,
    *,
    background: str = "#1D1C18",
) -> DotSphereFrame:
    """Ease between states without replacing the whole particle field at once."""

    if (previous.width, previous.height) != (current.width, current.height):
        raise ValueError("dot sphere frames must have matching dimensions")
    progress = max(0.0, min(1.0, progress))
    previous_cells = {(cell.x, cell.y): cell for cell in previous.cells}
    current_cells = {(cell.x, cell.y): cell for cell in current.cells}
    cells = []
    positions = previous_cells.keys() | current_cells.keys()
    for position in sorted(positions, key=lambda item: (item[1], item[0])):
        old = previous_cells.get(position)
        new = current_cells.get(position)
        if old is not None and new is not None:
            glyph = old.glyph if old.glyph == new.glyph or progress < 0.5 else new.glyph
            style = _blend_hex_color(old.style, new.style, progress)
            z = old.z + (new.z - old.z) * progress
            highlighted = new.highlighted if progress >= 0.5 else old.highlighted
        elif old is not None:
            glyph = old.glyph
            style = _blend_hex_color(old.style, background, progress)
            z = old.z
            highlighted = old.highlighted and progress < 0.5
        else:
            assert new is not None
            glyph = new.glyph
            style = _blend_hex_color(background, new.style, progress)
            z = new.z
            highlighted = new.highlighted and progress >= 0.5
        cells.append(
            DotCell(
                x=position[0],
                y=position[1],
                z=z,
                glyph=glyph,
                style=style,
                highlighted=highlighted,
            )
        )

    return DotSphereFrame(
        width=current.width,
        height=current.height,
        state=current.state,
        cells=tuple(cells),
    )
