"""The working-cloud renderer behind every sphere state but celebration.

One orbital particle sphere carries all stages; a state only changes the
overlay on that shared field (the extraction network while extracting,
highlighted particles while recalling), so stage changes read as continuous
motion instead of a redraw.
"""

from __future__ import annotations

import math

from everos.entrypoints.tui.demo.theme import (
    EVEROS_CYAN,
    EVEROS_YELLOW,
)
from everos.entrypoints.tui.demo.widgets.sphere._geometry import (
    _add_braille_dot,
    _cross_3d,
    _inside_sphere_projection,
    _normalize_3d,
    _particle_offsets_for_depth,
    _point_on_orbit,
    _replace_with_shared_outer_shell,
    _sphere_geometry,
    _stable_hash,
)
from everos.entrypoints.tui.demo.widgets.sphere._models import (
    BRAILLE_BASE,
    EXTRACT_BRANCH_COUNT,
    EXTRACT_TRAIL_GAP,
    EXTRACT_TRAIL_STEPS,
    GOLDEN_ANGLE,
    WORKING_MIN_ORBITS,
    WORKING_MIN_SAMPLES,
    WORKING_ORBITS_PER_RADIUS,
    WORKING_PARTICLES_PER_ORBIT,
    WORKING_SAMPLES_PER_RADIUS,
    DotCell,
    DotSphereFrame,
    SphereState,
)
from everos.entrypoints.tui.demo.widgets.sphere._styles import (
    _style_for_active_particle,
    _style_for_ghost_depth,
    _style_for_network_edge,
    _style_for_network_node,
    _style_for_network_signal,
    _style_for_network_surface,
    _style_for_shared_outer_shell,
)


def _build_working_cloud(
    *, width: int, height: int, phase: float, state: SphereState
) -> DotSphereFrame:
    """Render a full orbital sphere with state-specific white particles."""

    sub_width, sub_height, center_x, center_y, radius_x, radius_y = _sphere_geometry(
        width, height
    )
    animation_time = phase * math.tau
    orbit_count = max(
        WORKING_MIN_ORBITS,
        round(radius_x * WORKING_ORBITS_PER_RADIUS),
    )
    samples_per_orbit = max(
        WORKING_MIN_SAMPLES,
        round(radius_x * WORKING_SAMPLES_PER_RADIUS),
    )
    global_yaw = animation_time * 0.08
    camera_tilt = 0.18
    vertical_axis = (0.0, 1.0, 0.0)

    masks: dict[tuple[int, int], int] = {}
    depths: dict[tuple[int, int], float] = {}
    active_depths: dict[tuple[int, int], float] = {}
    for orbit in range(orbit_count):
        if orbit < 3:
            normal = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))[orbit]
        else:
            normal_y = 1 - 2 * ((orbit + 0.5) / orbit_count)
            normal_radius = math.sqrt(max(0.0, 1.0 - normal_y * normal_y))
            normal_theta = orbit * GOLDEN_ANGLE
            normal = (
                normal_radius * math.cos(normal_theta),
                normal_y,
                normal_radius * math.sin(normal_theta),
            )
        reference = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else vertical_axis
        basis_u = _normalize_3d(*_cross_3d(normal, reference))
        basis_v = _cross_3d(normal, basis_u)
        orbit_radius = (
            0.98
            if orbit < 3 or orbit % 4 == 0
            else 0.52 + 0.44 * _stable_hash(orbit, 2.7)
        )

        for sample in range(samples_per_orbit):
            angle = (sample / samples_per_orbit) * math.tau
            if orbit == 0:
                sub_x = round(center_x + math.cos(angle) * radius_x)
                sub_y = round(center_y - math.sin(angle) * radius_y)
                normalized_depth = math.sin(angle + global_yaw) * 0.35
            else:
                point = _point_on_orbit(
                    basis_u,
                    basis_v,
                    orbit_radius,
                    angle,
                    global_yaw,
                    camera_tilt,
                )
                sub_x = round(center_x + point[0] * radius_x)
                sub_y = round(center_y - point[1] * radius_y)
                normalized_depth = point[2] / orbit_radius
            if 0 <= sub_x < sub_width and 0 <= sub_y < sub_height:
                _add_braille_dot(
                    masks=masks,
                    depths=depths,
                    sub_x=sub_x,
                    sub_y=sub_y,
                    z=normalized_depth,
                )

        direction = 1 if orbit % 2 == 0 else -1
        speed = direction * (0.18 + 0.12 * _stable_hash(orbit, 7.3))
        for particle in range(WORKING_PARTICLES_PER_ORBIT):
            point = _point_on_orbit(
                basis_u,
                basis_v,
                orbit_radius,
                animation_time * speed
                + (particle / WORKING_PARTICLES_PER_ORBIT) * math.tau
                + _stable_hash(orbit, 5.1) * math.tau,
                global_yaw,
                camera_tilt,
            )
            sub_x = round(center_x + point[0] * radius_x)
            sub_y = round(center_y - point[1] * radius_y)
            normalized_depth = point[2] / orbit_radius
            if not (0 <= sub_x < sub_width and 0 <= sub_y < sub_height):
                continue
            for offset_x, offset_y in _particle_offsets_for_depth(normalized_depth):
                particle_x = sub_x + offset_x
                particle_y = sub_y + offset_y
                if not _inside_sphere_projection(
                    particle_x,
                    particle_y,
                    center_x,
                    center_y,
                    radius_x,
                    radius_y,
                ):
                    continue
                _add_braille_dot(
                    masks=masks,
                    depths=depths,
                    sub_x=particle_x,
                    sub_y=particle_y,
                    z=normalized_depth,
                )
                position = (particle_x // 2, particle_y // 4)
                active_depths[position] = max(
                    normalized_depth,
                    active_depths.get(position, -1.0),
                )

    shared_edge_positions = _replace_with_shared_outer_shell(
        masks=masks,
        depths=depths,
        layer_maps=(active_depths,),
        animation_time=animation_time,
        center_x=center_x,
        center_y=center_y,
        radius_x=radius_x,
        radius_y=radius_y,
    )

    network_edge_depths: dict[tuple[int, int], float] = {}
    network_edge_visibilities: dict[tuple[int, int], float] = {}
    network_node_depths: dict[tuple[int, int], float] = {}
    network_signal_depths: dict[tuple[int, int], float] = {}
    if state.key == "extracting":
        (
            network_edge_depths,
            network_edge_visibilities,
            network_node_depths,
            network_signal_depths,
        ) = _network_layers_on_particle_field(
            masks=masks,
            depths=depths,
            shared_edge_positions=shared_edge_positions,
            animation_time=animation_time,
            center_x=center_x,
            center_y=center_y,
            radius_x=radius_x,
            radius_y=radius_y,
        )

    highlighted_positions: set[tuple[int, int]] = set()
    if state.key in {"recalling", "remembered", "source"}:
        target_ratios = (
            ((0.72, 0.28), (0.63, 0.37), (0.78, 0.44), (0.57, 0.24))
            if state.key == "recalling"
            else ((0.72, 0.28),)
        )
        for target_x, target_y in target_ratios:
            available = (
                position
                for position in masks
                if position not in highlighted_positions
                and position not in shared_edge_positions
                and depths[position] > -0.15
            )
            highlighted_positions.add(
                min(
                    available,
                    key=lambda position: (
                        (position[0] - (width - 1) * target_x) ** 2
                        + (position[1] - (height - 1) * target_y) ** 2
                    ),
                )
            )

    cells = []
    for (x, y), mask in masks.items():
        position = (x, y)
        active_depth = active_depths.get((x, y))
        target_highlighted = (x, y) in highlighted_positions
        network_signal_depth = network_signal_depths.get(position)
        network_node_depth = network_node_depths.get(position)
        network_edge_depth = network_edge_depths.get(position)
        if position in shared_edge_positions:
            style = _style_for_shared_outer_shell(depths[(x, y)])
        elif network_signal_depth is not None:
            style = _style_for_network_signal(depths[position])
        elif network_node_depth is not None:
            style = _style_for_network_node(network_node_depth)
        elif network_edge_depth is not None:
            style = _style_for_network_edge(
                network_edge_depth,
                network_edge_visibilities[position],
            )
        elif target_highlighted and state.key == "recalling":
            style = EVEROS_CYAN
        elif target_highlighted:
            style = EVEROS_YELLOW
        elif state.key == "indexing" and depths[(x, y)] > 0.3:
            # Preserve the old Index behavior: the organized front layer turns
            # white, now projected onto the complete orbital sphere.
            style = EVEROS_CYAN
        elif active_depth is not None and state.key == "ingesting":
            style = _style_for_active_particle(active_depth, allow_white=True)
        elif state.key == "extracting":
            style = _style_for_network_surface(depths[position])
        elif active_depth is not None:
            style = _style_for_active_particle(active_depth, allow_white=False)
        else:
            style = _style_for_ghost_depth(depths[(x, y)])
        highlighted = (
            target_highlighted
            or network_signal_depth is not None
            or style == EVEROS_CYAN
        )
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


def _network_layers_on_particle_field(
    *,
    masks: dict[tuple[int, int], int],
    depths: dict[tuple[int, int], float],
    shared_edge_positions: set[tuple[int, int]],
    animation_time: float,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """Color a moving network onto the shared particle field.

    Extract used to build an independent sphere, so entering the stage replaced
    most of the center in one frame. This layer snaps its graph, nodes, and
    moving packets to particles that already exist in every stage. The shape
    therefore remains continuous while the color animation still reads as a
    connected extraction network.
    """

    candidates = tuple(
        position for position in masks if position not in shared_edge_positions
    )
    candidate_set = set(candidates)
    edge_depths: dict[tuple[int, int], float] = {}
    edge_visibilities: dict[tuple[int, int], float] = {}
    node_depths: dict[tuple[int, int], float] = {}
    signal_depths: dict[tuple[int, int], float] = {}
    if not candidates:
        return edge_depths, edge_visibilities, node_depths, signal_depths

    def nearest_particle(
        sub_x: float,
        sub_y: float,
        depth: float,
    ) -> tuple[int, int]:
        depth_scale = min(radius_x, radius_y) * 0.22
        target_x = round((sub_x - 0.5) / 2)
        target_y = round((sub_y - 1.5) / 4)
        local_candidates = tuple(
            position
            for y in range(target_y - 2, target_y + 3)
            for x in range(target_x - 3, target_x + 4)
            if (position := (x, y)) in candidate_set
        )
        return min(
            local_candidates or candidates,
            key=lambda position: (
                (position[0] * 2 + 0.5 - sub_x) ** 2
                + (position[1] * 4 + 1.5 - sub_y) ** 2
                + ((depths[position] - depth) * depth_scale) ** 2
            ),
        )

    branch_rotation = 0.07 * math.sin(animation_time * 0.18)

    def branch_point(
        branch: int,
        progress: float,
    ) -> tuple[float, float, float]:
        base_angle = branch * math.tau / EXTRACT_BRANCH_COUNT - math.pi / 2
        bend = (1 if branch % 2 == 0 else -1) * 0.24
        angle = base_angle + branch_rotation + bend * math.sin(progress * math.pi)
        branch_length = 0.58 + 0.07 * _stable_hash(branch, 71.3)
        radial = 0.06 + branch_length * progress
        depth_limit = math.sqrt(max(0.0, 1 - radial * radial))
        depth = (
            math.cos(base_angle + animation_time * 0.1)
            * depth_limit
            * (0.35 + 0.55 * progress)
        )
        return (
            center_x + math.cos(angle) * radius_x * radial,
            center_y - math.sin(angle) * radius_y * radial,
            depth,
        )

    samples_per_branch = max(12, round(radius_x * 0.55))
    for branch in range(EXTRACT_BRANCH_COUNT):
        for sample in range(samples_per_branch + 1):
            progress = sample / samples_per_branch
            sub_x, sub_y, depth = branch_point(branch, progress)
            position = nearest_particle(sub_x, sub_y, depth)
            edge_depths[position] = max(depth, edge_depths.get(position, -1.0))
            edge_visibilities[position] = max(
                0.2 + 0.28 * progress,
                edge_visibilities.get(position, 0.0),
            )

        for node_progress in (0.06, 0.5, 1.0):
            sub_x, sub_y, depth = branch_point(branch, node_progress)
            position = nearest_particle(sub_x, sub_y, depth)
            node_depths[position] = max(depth, node_depths.get(position, -1.0))

        head_progress = (animation_time * 0.16 + branch / EXTRACT_BRANCH_COUNT) % 1.0
        for trail_step in range(EXTRACT_TRAIL_STEPS):
            progress = head_progress - trail_step * EXTRACT_TRAIL_GAP
            if progress < 0:
                continue
            sub_x, sub_y, depth = branch_point(branch, progress)
            position = nearest_particle(sub_x, sub_y, depth)
            signal_depths[position] = max(
                depth,
                signal_depths.get(position, -1.0),
            )

    source = nearest_particle(center_x, center_y, 0.85)
    node_depths[source] = 0.85

    return edge_depths, edge_visibilities, node_depths, signal_depths
