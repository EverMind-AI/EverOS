"""EverOS demo dot-sphere rendering contracts."""

from __future__ import annotations

import math

from everos.entrypoints.tui.demo.widgets.sphere import (
    EXTRACT_BRANCH_COUNT,
    SHARED_EDGE_INNER_RADIUS,
    SPHERE_STATES,
    DotSphereFrame,
    _cell_projection_radius,
    _sphere_geometry,
    blend_dot_sphere_frames,
    build_dot_sphere,
)


def test_pipeline_states_form_complete_particle_spheres() -> None:
    for state_key in ("ingesting", "extracting", "indexing", "recalling"):
        frame = build_dot_sphere(
            width=41,
            height=19,
            phase=0.0,
            state_key=state_key,
        )

        assert frame.width == 41
        assert frame.height == 19
        assert len(frame.cells) >= 300

        center_x = (frame.width - 1) / 2
        center_y = (frame.height - 1) / 2
        radius_x = center_x
        radius_y = center_y
        for cell in frame.cells:
            normalized = ((cell.x - center_x) / radius_x) ** 2 + (
                (cell.y - center_y) / radius_y
            ) ** 2
            assert normalized <= 1.08

        center_cells = [
            cell
            for cell in frame.cells
            if abs(cell.x - center_x) < frame.width * 0.08
            and abs(cell.y - center_y) < frame.height * 0.12
        ]
        assert len(center_cells) >= 20


def test_dot_sphere_keeps_complete_sphere_inside_terminal_frame() -> None:
    frame = build_dot_sphere(width=37, height=17, phase=0.0, state_key="indexing")
    row_spans = _row_spans(frame)

    occupied_rows = [y for y, span in row_spans.items() if span]
    assert min(occupied_rows) >= 1
    assert max(occupied_rows) <= frame.height - 2
    assert row_spans[frame.height // 2] >= 28


def test_all_pipeline_states_keep_a_round_outer_shell_through_cycle() -> None:
    for state_key in ("ingesting", "extracting", "indexing", "recalling"):
        occupied_widths = []
        occupied_heights = []
        for phase in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
            frame = build_dot_sphere(
                width=37,
                height=17,
                phase=phase,
                state_key=state_key,
            )
            xs = [cell.x for cell in frame.cells]
            ys = [cell.y for cell in frame.cells]
            occupied_width = max(xs) - min(xs) + 1
            occupied_height = max(ys) - min(ys) + 1
            physical_height = occupied_height * 2
            occupied_widths.append(occupied_width)
            occupied_heights.append(occupied_height)

            assert 0.93 <= occupied_width / physical_height <= 1.08

        assert max(occupied_widths) - min(occupied_widths) <= 2
        assert max(occupied_heights) - min(occupied_heights) <= 1


def test_processing_states_match_the_unchanged_particle_density() -> None:
    reference = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="indexing",
    )
    reference_density = sum(
        _braille_subdot_count(cell.glyph) for cell in reference.cells
    )

    for state_key in ("booting", "ingesting", "extracting"):
        frame = build_dot_sphere(
            width=41,
            height=19,
            phase=0.25,
            state_key=state_key,
        )
        density = sum(_braille_subdot_count(cell.glyph) for cell in frame.cells)

        assert reference_density * 0.9 <= density <= reference_density * 1.15


def test_processing_particle_density_adapts_to_terminal_size() -> None:
    for state_key in ("ingesting", "extracting", "indexing", "recalling"):
        dot_counts = []
        for width, height in ((25, 11), (41, 19), (61, 27)):
            frame = build_dot_sphere(
                width=width,
                height=height,
                phase=0.25,
                state_key=state_key,
            )
            dot_counts.append(
                sum(_braille_subdot_count(cell.glyph) for cell in frame.cells)
            )

        assert dot_counts == sorted(dot_counts)
        assert dot_counts[-1] > dot_counts[0] * 3


def test_processing_states_share_the_same_adaptive_outer_size() -> None:
    for width, height in ((25, 11), (37, 17), (41, 19), (61, 27)):
        reference = build_dot_sphere(
            width=width,
            height=height,
            phase=0.25,
            state_key="ingesting",
        )
        for state_key in ("extracting", "indexing", "recalling"):
            frame = build_dot_sphere(
                width=width,
                height=height,
                phase=0.25,
                state_key=state_key,
            )
            assert _frame_extents(reference) == _frame_extents(frame)


def test_pipeline_states_keep_the_same_outer_position_through_cycle() -> None:
    for phase in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
        reference = build_dot_sphere(
            width=41,
            height=19,
            phase=phase,
            state_key="ingesting",
        )
        for state_key in ("extracting", "indexing", "recalling"):
            frame = build_dot_sphere(
                width=41,
                height=19,
                phase=phase,
                state_key=state_key,
            )
            assert _frame_extents(reference) == _frame_extents(frame)


def test_pipeline_states_share_the_exact_same_particle_edge() -> None:
    _, _, center_x, center_y, radius_x, radius_y = _sphere_geometry(41, 19)

    for phase in (0.0, 0.25, 0.5, 0.75):
        edges = []
        for state_key in ("ingesting", "extracting", "indexing", "recalling"):
            frame = build_dot_sphere(
                width=41,
                height=19,
                phase=phase,
                state_key=state_key,
            )
            edges.append(
                {
                    (cell.x, cell.y, cell.glyph, cell.style)
                    for cell in frame.cells
                    if _cell_projection_radius(
                        (cell.x, cell.y),
                        center_x=center_x,
                        center_y=center_y,
                        radius_x=radius_x,
                        radius_y=radius_y,
                    )
                    >= SHARED_EDGE_INNER_RADIUS
                }
            )

        assert all(edge == edges[0] for edge in edges[1:])


def test_pipeline_states_share_the_exact_same_particle_field() -> None:
    for phase in (0.0, 0.25, 0.5, 0.75):
        reference = build_dot_sphere(
            width=41,
            height=19,
            phase=phase,
            state_key="ingesting",
        )
        reference_geometry = [
            (cell.x, cell.y, cell.glyph, cell.z) for cell in reference.cells
        ]

        for state_key in ("extracting", "indexing", "recalling"):
            frame = build_dot_sphere(
                width=41,
                height=19,
                phase=phase,
                state_key=state_key,
            )
            assert [
                (cell.x, cell.y, cell.glyph, cell.z) for cell in frame.cells
            ] == reference_geometry


def test_pipeline_state_color_transition_preserves_particle_geometry() -> None:
    ingesting = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="ingesting",
    )
    extracting = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="extracting",
    )
    blended = blend_dot_sphere_frames(ingesting, extracting, 0.5)

    assert [(cell.x, cell.y, cell.glyph) for cell in blended.cells] == [
        (cell.x, cell.y, cell.glyph) for cell in ingesting.cells
    ]
    assert {cell.style for cell in blended.cells} != {
        cell.style for cell in ingesting.cells
    }
    assert {cell.style for cell in blended.cells} != {
        cell.style for cell in extracting.cells
    }


def test_shared_particle_edge_moves_without_density_or_position_jumps() -> None:
    _, _, center_x, center_y, radius_x, radius_y = _sphere_geometry(41, 19)

    edge_frames = []
    for phase in (0.25, 0.2625, 0.5):
        frame = build_dot_sphere(
            width=41,
            height=19,
            phase=phase,
            state_key="ingesting",
        )
        edge_frames.append(
            {
                (cell.x, cell.y, cell.glyph)
                for cell in frame.cells
                if _cell_projection_radius(
                    (cell.x, cell.y),
                    center_x=center_x,
                    center_y=center_y,
                    radius_x=radius_x,
                    radius_y=radius_y,
                )
                >= SHARED_EDGE_INNER_RADIUS
            }
        )

    assert edge_frames[0] != edge_frames[1]
    assert len(edge_frames[0] ^ edge_frames[2]) > len(edge_frames[0]) * 0.2
    for index in range(len(edge_frames) - 1):
        before = edge_frames[index]
        after = edge_frames[index + 1]
        assert abs(len(after) - len(before)) < len(before) * 0.06
        assert (
            _position_hausdorff_distance(
                {(x, y) for x, y, _ in before},
                {(x, y) for x, y, _ in after},
            )
            <= 1
        )


def test_pipeline_states_keep_comparable_default_size_density() -> None:
    for phase in (0.0, 0.5, 1.0, 1.5, 1.8):
        reference = build_dot_sphere(
            width=37,
            height=17,
            phase=phase,
            state_key="ingesting",
        )
        reference_density = sum(
            _braille_subdot_count(cell.glyph) for cell in reference.cells
        )
        for state_key in ("extracting", "indexing", "recalling"):
            frame = build_dot_sphere(
                width=37,
                height=17,
                phase=phase,
                state_key=state_key,
            )
            density = sum(_braille_subdot_count(cell.glyph) for cell in frame.cells)
            assert reference_density * 0.88 <= density
            assert density <= reference_density * 1.12


def test_working_reference_uses_dark_paths_and_bright_moving_particles() -> None:
    frame = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="booting",
    )
    dark_styles = {"#61522F", "#76612F", "#8C6D2B", "#A97D25"}
    bright_styles = {"#DDA21E", "#F9B91C", "#FFD267"}
    dark_cells = [cell for cell in frame.cells if cell.style in dark_styles]

    assert len(dark_cells) > len(frame.cells) * 0.75
    assert bright_styles <= {cell.style for cell in frame.cells}


def test_processing_states_make_near_particles_larger_than_far_particles() -> None:
    for state_key in ("ingesting", "extracting", "indexing", "recalling"):
        frame = build_dot_sphere(
            width=41,
            height=19,
            phase=0.25,
            state_key=state_key,
        )
        near = [cell for cell in frame.cells if cell.z > 0.6]
        far = [cell for cell in frame.cells if cell.z < -0.6]
        near_size = sum(_braille_subdot_count(cell.glyph) for cell in near) / len(near)
        far_size = sum(_braille_subdot_count(cell.glyph) for cell in far) / len(far)

        assert near_size > far_size * 1.35
        assert {"#FFD267", "#F5EDDC"} & {cell.style for cell in near}
        assert "#F5EDDC" not in {cell.style for cell in far}


def test_processing_particles_move_without_density_jumps() -> None:
    for state_key in ("ingesting", "extracting"):
        before = build_dot_sphere(
            width=41,
            height=19,
            phase=0.25,
            state_key=state_key,
        )
        after = build_dot_sphere(
            width=41,
            height=19,
            phase=0.3,
            state_key=state_key,
        )
        before_active = {(cell.x, cell.y) for cell in before.cells if cell.highlighted}
        after_active = {(cell.x, cell.y) for cell in after.cells if cell.highlighted}
        before_density = sum(_braille_subdot_count(cell.glyph) for cell in before.cells)
        after_density = sum(_braille_subdot_count(cell.glyph) for cell in after.cells)

        assert before_active != after_active
        assert abs(before_density - after_density) < before_density * 0.08


def test_dot_sphere_uses_braille_fine_dot_cells() -> None:
    for state_key in SPHERE_STATES:
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
    frame = build_dot_sphere(width=37, height=17, phase=0.0, state_key="indexing")

    subdot_count = sum(_braille_subdot_count(cell.glyph) for cell in frame.cells)

    assert subdot_count > len(frame.cells) * 1.8
    assert subdot_count > frame.width * frame.height * 0.65
    assert any(_braille_subdot_count(cell.glyph) >= 4 for cell in frame.cells)


def test_dot_sphere_breathes_between_animation_phases() -> None:
    frame = build_dot_sphere(width=37, height=17, phase=0.0, state_key="booting")
    next_frame = build_dot_sphere(width=37, height=17, phase=0.125, state_key="booting")

    assert {(cell.x, cell.y, cell.glyph) for cell in frame.cells} != {
        (cell.x, cell.y, cell.glyph) for cell in next_frame.cells
    }


def test_dot_sphere_stays_continuous_past_old_phase_wrap() -> None:
    before = build_dot_sphere(
        width=41,
        height=19,
        phase=0.9875,
        state_key="extracting",
    )
    after = build_dot_sphere(
        width=41,
        height=19,
        phase=1.0,
        state_key="extracting",
    )
    wrapped = build_dot_sphere(
        width=41,
        height=19,
        phase=0.0,
        state_key="extracting",
    )
    before_positions = {(cell.x, cell.y) for cell in before.cells}
    after_positions = {(cell.x, cell.y) for cell in after.cells}
    wrapped_positions = {(cell.x, cell.y) for cell in wrapped.cells}

    assert len(before_positions ^ after_positions) < len(
        before_positions ^ wrapped_positions
    )


def test_pipeline_states_keep_a_filled_center_during_animation_cycle() -> None:
    for state_key in ("ingesting", "extracting", "indexing", "recalling"):
        for phase in (0.0, 0.25, 0.5, 0.75):
            frame = build_dot_sphere(
                width=41,
                height=19,
                phase=phase,
                state_key=state_key,
            )
            center_x = (frame.width - 1) / 2
            center_y = (frame.height - 1) / 2

            center_cells = [
                cell
                for cell in frame.cells
                if abs(cell.x - center_x) < frame.width * 0.08
                and abs(cell.y - center_y) < frame.height * 0.12
            ]
            assert len(center_cells) >= 20


def test_working_and_ingesting_share_particle_motion_but_not_color() -> None:
    working = build_dot_sphere(width=41, height=19, phase=0.25, state_key="booting")
    ingesting = build_dot_sphere(width=41, height=19, phase=0.25, state_key="ingesting")

    assert [(cell.x, cell.y, cell.glyph) for cell in working.cells] == [
        (cell.x, cell.y, cell.glyph) for cell in ingesting.cells
    ]
    assert len(working.cells) >= 190
    assert "#F5EDDC" not in {cell.style for cell in working.cells}
    assert "#F5EDDC" in {cell.style for cell in ingesting.cells}
    assert working.caption == "working..."


def test_extracting_uses_outward_branches_with_internal_sparks() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.25, state_key="extracting")
    center_x = (frame.width - 1) / 2
    center_y = (frame.height - 1) / 2

    assert any(
        abs(cell.x - center_x) < frame.width * 0.08
        and abs(cell.y - center_y) < frame.height * 0.12
        for cell in frame.cells
    )
    highlighted = [cell for cell in frame.cells if cell.highlighted]
    assert len(frame.cells) >= 120
    assert len(highlighted) >= EXTRACT_BRANCH_COUNT
    highlighted_styles = {cell.style for cell in highlighted}
    assert "#F5EDDC" in highlighted_styles
    assert highlighted_styles <= {
        "#76612F",
        "#8C6D2B",
        "#A97D25",
        "#C48E20",
        "#DDA21E",
        "#F9B91C",
        "#FFD267",
        "#F5EDDC",
    }
    assert "#FFD267" in {cell.style for cell in frame.cells}


def test_extracting_signals_follow_branches_without_teleporting() -> None:
    for phase in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        before = build_dot_sphere(
            width=41,
            height=19,
            phase=phase,
            state_key="extracting",
        )
        after = build_dot_sphere(
            width=41,
            height=19,
            phase=phase + 0.002,
            state_key="extracting",
        )
        before_signals = {(cell.x, cell.y) for cell in before.cells if cell.highlighted}
        after_signals = {(cell.x, cell.y) for cell in after.cells if cell.highlighted}

        assert _position_hausdorff_distance(before_signals, after_signals) <= 2


def test_extracting_uses_color_depth_without_changing_geometry() -> None:
    frame = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="extracting",
    )
    color_rank = {
        "#61522F": 0,
        "#76612F": 1,
        "#8C6D2B": 2,
        "#A97D25": 3,
        "#C48E20": 4,
        "#DDA21E": 5,
        "#F9B91C": 6,
        "#FFD267": 7,
    }
    front = [cell for cell in frame.cells if cell.z > 0.45 and not cell.highlighted]
    back = [cell for cell in frame.cells if cell.z < -0.45 and not cell.highlighted]
    front_brightness = sum(color_rank[cell.style] for cell in front) / len(front)
    back_brightness = sum(color_rank[cell.style] for cell in back) / len(back)

    assert front_brightness > back_brightness + 3
    ingesting = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="ingesting",
    )
    assert _frame_extents(frame) == _frame_extents(ingesting)


def test_extracting_white_signal_visits_front_and_side() -> None:
    white_depths = []
    for phase in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
        frame = build_dot_sphere(
            width=41,
            height=19,
            phase=phase,
            state_key="extracting",
        )
        white_depths.extend(
            cell.z
            for cell in frame.cells
            if cell.highlighted and cell.style == "#F5EDDC"
        )

    assert any(depth > 0.35 for depth in white_depths)
    assert any(-0.32 < depth < 0.25 for depth in white_depths)


def test_dot_sphere_remembered_state_has_highlighted_node() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.25, state_key="remembered")

    highlighted = [cell for cell in frame.cells if cell.highlighted]
    assert len(highlighted) == 1
    assert _is_braille_cell(highlighted[0].glyph)
    assert highlighted[0].style == "#F9B91C"
    assert frame.caption == "found the matching memory"


def test_celebrating_supernova_seeds_the_center_and_restores_the_sphere() -> None:
    recalled = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="recalling",
    )
    start = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="celebrating",
        state_phase=0.0,
    )
    burst = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="celebrating",
        state_phase=0.06,
    )
    scattered = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="celebrating",
        state_phase=0.16,
    )
    drifted = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="celebrating",
        state_phase=0.2,
    )
    post_burst_blank = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="celebrating",
        state_phase=0.3,
    )
    core_frames = [
        build_dot_sphere(
            width=41,
            height=19,
            phase=0.93,
            state_key="celebrating",
            state_phase=core_phase,
        )
        for core_phase in (0.4, 0.5, 0.64)
    ]
    emerging = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="celebrating",
        state_phase=0.72,
    )
    restored = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93,
        state_key="celebrating",
        state_phase=1.0,
    )
    next_ingest = build_dot_sphere(
        width=41,
        height=19,
        phase=0.93 + 2.4,
        state_key="ingesting",
    )

    assert start.caption == "memory crystallized"
    assert [(cell.x, cell.y, cell.glyph, cell.style) for cell in start.cells] == [
        (cell.x, cell.y, cell.glyph, cell.style) for cell in recalled.cells
    ]
    assert all(_is_braille_cell(cell.glyph) for cell in burst.cells)
    assert not any(cell.glyph in {"*", "+", ".", "x"} for cell in burst.cells)

    center_x = (scattered.width - 1) / 2
    center_y = (scattered.height - 1) / 2

    def mean_radius(frame: DotSphereFrame) -> float:
        return sum(
            math.hypot(cell.x - center_x, (cell.y - center_y) * 2)
            for cell in frame.cells
        ) / len(frame.cells)

    assert mean_radius(scattered) > mean_radius(start) * 1.14
    start_dots = sum(_braille_subdot_count(cell.glyph) for cell in start.cells)
    burst_dots = sum(_braille_subdot_count(cell.glyph) for cell in burst.cells)
    assert burst_dots > start_dots * 1.12
    for exploding_frame in (burst, scattered, drifted):
        border_cells = sum(
            cell.x in {0, 40} or cell.y in {0, 18} for cell in exploding_frame.cells
        )
        assert border_cells == 0
        corner_cells = sum(
            abs(cell.x - center_x) > 14 and abs(cell.y - center_y) > 6
            for cell in exploding_frame.cells
        )
        assert corner_cells < len(exploding_frame.cells) * 0.04
    assert [(cell.x, cell.y, cell.glyph) for cell in drifted.cells] != [
        (cell.x, cell.y, cell.glyph) for cell in scattered.cells
    ]
    assert not post_burst_blank.cells
    for core in core_frames:
        assert 0 < len(core.cells) < len(start.cells) * 0.2
        assert all(
            math.hypot(cell.x - center_x, (cell.y - center_y) * 2) < 6.5
            for cell in core.cells
        )
    core_styles = {cell.style for cell in core_frames[1].cells}
    assert "#F5EDDC" in core_styles
    assert core_styles - {"#F5EDDC", "#24231E"}
    assert [(cell.x, cell.y, cell.glyph) for cell in core_frames[1].cells] != [
        (cell.x, cell.y, cell.glyph) for cell in core_frames[2].cells
    ]
    assert 0 < len(emerging.cells) < len(restored.cells)
    assert [(cell.x, cell.y, cell.glyph, cell.style) for cell in restored.cells] == [
        (cell.x, cell.y, cell.glyph, cell.style) for cell in next_ingest.cells
    ]


def test_dot_sphere_front_light_uses_poster_gold_primary() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.0, state_key="booting")

    front_styles = {cell.style for cell in frame.cells if cell.z > 0.05}
    assert "#F9B91C" in front_styles
    assert "bold #F9B91C" not in front_styles
    assert "#FFE600" not in front_styles


def test_dot_sphere_uses_eight_gold_depth_levels() -> None:
    frame = build_dot_sphere(width=41, height=19, phase=0.25, state_key="booting")
    styles = {cell.style for cell in frame.cells if not cell.highlighted}

    assert styles == {
        "#61522F",
        "#76612F",
        "#8C6D2B",
        "#A97D25",
        "#C48E20",
        "#DDA21E",
        "#F9B91C",
        "#FFD267",
    }
    assert "#F5EDDC" not in styles


def test_dot_sphere_preserves_state_specific_white_and_gold_effects() -> None:
    ingesting = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="ingesting",
    )
    indexing = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="indexing",
    )
    extracting = build_dot_sphere(
        width=41,
        height=19,
        phase=0.25,
        state_key="extracting",
    )

    assert "#F5EDDC" in {cell.style for cell in ingesting.cells}
    assert "#F5EDDC" in {cell.style for cell in indexing.cells}
    extracting_styles = {cell.style for cell in extracting.cells}
    assert "#F5EDDC" in extracting_styles
    assert "#FFD267" in extracting_styles
    assert "#C09525" not in extracting_styles


def test_ingesting_and_indexing_use_different_white_proportions() -> None:
    ingesting = build_dot_sphere(width=41, height=19, phase=0.25, state_key="ingesting")
    indexing = build_dot_sphere(width=41, height=19, phase=0.25, state_key="indexing")

    ingesting_white = sum(cell.style == "#F5EDDC" for cell in ingesting.cells)
    indexing_white = sum(cell.style == "#F5EDDC" for cell in indexing.cells)
    assert indexing_white > ingesting_white * 1.5

    allowed_styles = {
        "#61522F",
        "#76612F",
        "#8C6D2B",
        "#A97D25",
        "#C48E20",
        "#DDA21E",
        "#F9B91C",
        "#FFD267",
        "#F5EDDC",
    }
    assert {cell.style for cell in ingesting.cells} <= allowed_styles
    assert {cell.style for cell in indexing.cells} <= allowed_styles


def test_recalling_highlights_several_white_memory_nodes() -> None:
    for phase in (0.0, 0.5, 1.0, 1.5, 2.0):
        frame = build_dot_sphere(
            width=41,
            height=19,
            phase=phase,
            state_key="recalling",
        )

        highlighted = [cell for cell in frame.cells if cell.highlighted]
        assert len(highlighted) == 4
        assert {cell.style for cell in highlighted} == {"#F5EDDC"}


def _row_spans(frame: DotSphereFrame) -> dict[int, int]:
    spans: dict[int, int] = {}
    for y in range(frame.height):
        xs = [cell.x for cell in frame.cells if cell.y == y]
        spans[y] = max(xs) - min(xs) + 1 if xs else 0
    return spans


def _frame_extents(frame: DotSphereFrame) -> tuple[int, int, int, int]:
    xs = [cell.x for cell in frame.cells]
    ys = [cell.y for cell in frame.cells]
    return min(xs), max(xs), min(ys), max(ys)


def _position_hausdorff_distance(
    before: set[tuple[int, int]],
    after: set[tuple[int, int]],
) -> int:
    def directed(
        source: set[tuple[int, int]],
        target: set[tuple[int, int]],
    ) -> int:
        return max(
            min(
                max(abs(x - target_x), abs(y - target_y))
                for target_x, target_y in target
            )
            for x, y in source
        )

    return max(directed(before, after), directed(after, before))


def _is_braille_cell(glyph: str) -> bool:
    return len(glyph) == 1 and 0x2800 < ord(glyph) <= 0x28FF


def _braille_subdot_count(glyph: str) -> int:
    assert _is_braille_cell(glyph)
    return (ord(glyph) - 0x2800).bit_count()
