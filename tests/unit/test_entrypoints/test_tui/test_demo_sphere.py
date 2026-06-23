"""EverOS demo dot-sphere rendering contracts."""

from __future__ import annotations

from everos.entrypoints.tui.demo.widgets.sphere import (
    SPHERE_STATES,
    DotSphereFrame,
    build_dot_sphere,
)


def test_dot_sphere_forms_round_bounded_cloud() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.0, state_key="extracting")

    assert frame.width == 41
    assert frame.height == 19
    assert frame.caption == "extracting episode -> atomic facts"
    assert len(frame.cells) >= 90

    center_x = (frame.width - 1) / 2
    center_y = (frame.height - 1) / 2
    radius_x = center_x
    radius_y = center_y
    for cell in frame.cells:
        normalized = ((cell.x - center_x) / radius_x) ** 2 + (
            (cell.y - center_y) / radius_y
        ) ** 2
        assert normalized <= 1.08

    row_counts: dict[int, int] = {}
    for cell in frame.cells:
        row_counts[cell.y] = row_counts.get(cell.y, 0) + 1
    assert row_counts[frame.height // 2] > row_counts[min(row_counts)]
    assert row_counts[frame.height // 2] > row_counts[max(row_counts)]


def test_dot_sphere_keeps_terminal_poles_visually_round() -> None:
    frame = build_dot_sphere(width=37, height=17, phase=0.0, state_key="booting")
    row_spans = _row_spans(frame)

    assert row_spans[0] <= 8
    assert row_spans[1] <= 20
    assert row_spans[2] <= 26
    assert row_spans[frame.height // 2] >= 31
    assert row_spans[frame.height - 2] <= 20
    assert row_spans[frame.height - 1] <= 8


def test_dot_sphere_uses_braille_fine_dot_cells() -> None:
    for state_key in SPHERE_STATES:
        if state_key == "celebrating":
            continue
        frame = build_dot_sphere(
            width=37,
            height=17,
            phase=0.25,
            state_key=state_key,
        )

        assert all(_is_braille_cell(cell.glyph) for cell in frame.cells)
        assert not any(cell.glyph in {"·", "•", "●", "◆"} for cell in frame.cells)
        assert not any(cell.style.startswith("bold ") for cell in frame.cells)


def test_dot_sphere_packs_multiple_subdots_per_terminal_cell() -> None:
    frame = build_dot_sphere(width=37, height=17, phase=0.0, state_key="booting")

    subdot_count = sum(_braille_subdot_count(cell.glyph) for cell in frame.cells)

    assert subdot_count > len(frame.cells) * 1.8
    assert subdot_count > frame.width * frame.height * 0.9
    assert any(_braille_subdot_count(cell.glyph) >= 4 for cell in frame.cells)


def test_dot_sphere_avoids_flat_sides_in_terminal_frame() -> None:
    frame = build_dot_sphere(width=37, height=17, phase=0.0, state_key="booting")
    row_spans = _row_spans(frame)

    assert max(row_spans.values()) <= frame.width - 4
    assert row_spans[frame.height // 2] >= frame.width - 4
    for y in range(frame.height // 2):
        assert abs(row_spans[y] - row_spans[frame.height - 1 - y]) <= 4


def test_dot_sphere_remembered_state_has_highlighted_node() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.25, state_key="remembered")

    highlighted = [cell for cell in frame.cells if cell.highlighted]
    assert len(highlighted) == 1
    assert _is_braille_cell(highlighted[0].glyph)
    assert highlighted[0].style == "#F9B91C"
    assert frame.caption == "remembered Yosemite preference"


def test_dot_sphere_celebrating_state_bursts_into_confetti() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.93, state_key="celebrating")

    assert frame.caption == "memory crystallized"
    confetti = [cell for cell in frame.cells if cell.glyph in {"*", "+", ".", "x"}]
    assert len(confetti) >= 70
    assert all(not _is_braille_cell(cell.glyph) for cell in confetti)
    assert not any(cell.style.startswith("bold ") for cell in confetti)

    center_x = (frame.width - 1) / 2
    center_y = (frame.height - 1) / 2
    radius_x = max(1.0, center_x - 3)
    radius_y = max(1.0, center_y - 2)
    distances = [
        ((cell.x - center_x) / radius_x) ** 2 + ((cell.y - center_y) / radius_y) ** 2
        for cell in confetti
    ]
    assert max(distances) > 1.10
    assert sum(distance > 0.72 for distance in distances) > len(distances) * 0.4

    styles = {cell.style for cell in confetti}
    assert "#F9B91C" in styles
    assert "#F6C23B" in styles


def test_dot_sphere_front_light_uses_poster_gold_primary() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.0, state_key="booting")

    front_styles = {cell.style for cell in frame.cells if cell.z > 0.05}
    assert "#F9B91C" in front_styles
    assert "bold #F9B91C" not in front_styles
    assert "#FFE600" not in front_styles


def _row_spans(frame: DotSphereFrame) -> dict[int, int]:
    spans: dict[int, int] = {}
    for y in range(frame.height):
        xs = [cell.x for cell in frame.cells if cell.y == y]
        spans[y] = max(xs) - min(xs) + 1 if xs else 0
    return spans


def _is_braille_cell(glyph: str) -> bool:
    return len(glyph) == 1 and 0x2800 < ord(glyph) <= 0x28FF


def _braille_subdot_count(glyph: str) -> int:
    assert _is_braille_cell(glyph)
    return (ord(glyph) - 0x2800).bit_count()
