"""EverOS demo dot-sphere rendering contracts."""

from __future__ import annotations

from everos.entrypoints.cli.demo_sphere import build_dot_sphere


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


def test_dot_sphere_remembered_state_has_highlighted_node() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.25, state_key="remembered")

    highlighted = [cell for cell in frame.cells if cell.highlighted]
    assert len(highlighted) == 1
    assert highlighted[0].glyph == "◆"
    assert highlighted[0].style == "bold #FFE600"
    assert frame.caption == "remembered Yosemite preference"


def test_dot_sphere_front_light_uses_bright_yellow_primary() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.0, state_key="booting")

    front_styles = {cell.style for cell in frame.cells if cell.z > 0.05}
    assert "#FFE600" in front_styles
    assert "bold #FFE600" in front_styles
