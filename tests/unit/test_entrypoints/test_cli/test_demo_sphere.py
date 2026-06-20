"""EverOS demo dot-sphere rendering contracts."""

from __future__ import annotations

from everos.entrypoints.cli.demo_sphere import DotSphereFrame, build_dot_sphere


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
    frame = build_dot_sphere(width=35, height=17, phase=0.0, state_key="booting")
    row_spans = _row_spans(frame)

    assert row_spans[0] == 1
    assert row_spans[1] <= 12
    assert row_spans[2] <= 20
    assert row_spans[frame.height // 2] >= 33
    assert row_spans[frame.height - 2] <= 12
    assert row_spans[frame.height - 1] == 1


def test_dot_sphere_remembered_state_has_highlighted_node() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.25, state_key="remembered")

    highlighted = [cell for cell in frame.cells if cell.highlighted]
    assert len(highlighted) == 1
    assert highlighted[0].glyph == "◆"
    assert highlighted[0].style == "bold #F9B91C"
    assert frame.caption == "remembered Yosemite preference"


def test_dot_sphere_front_light_uses_poster_gold_primary() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.0, state_key="booting")

    front_styles = {cell.style for cell in frame.cells if cell.z > 0.05}
    assert "#F9B91C" in front_styles
    assert "bold #F9B91C" in front_styles
    assert "#FFE600" not in front_styles


def _row_spans(frame: DotSphereFrame) -> dict[int, int]:
    spans: dict[int, int] = {}
    for y in range(frame.height):
        xs = [cell.x for cell in frame.cells if cell.y == y]
        spans[y] = max(xs) - min(xs) + 1 if xs else 0
    return spans
