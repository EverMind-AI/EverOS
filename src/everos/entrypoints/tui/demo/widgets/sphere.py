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
EVEROS_YELLOW_PALE = "#FFD267"
EVEROS_AMBER_DIM = "#4A3D20"
EVEROS_AMBER = "#8B763F"
EVEROS_GOLD_SHADOW = "#61522F"
EVEROS_GOLD_DEEP = "#76612F"
EVEROS_GOLD_DARK = "#8C6D2B"
EVEROS_GOLD_MID = "#A97D25"
EVEROS_GOLD_WARM = "#C48E20"
EVEROS_GOLD_LIGHT = "#DDA21E"
EVEROS_CYAN = "#F5EDDC"
EVEROS_GREEN = "#D8CDAF"
EVEROS_ORANGE = "#C09525"
EVEROS_FIELD_BACKGROUND = "#24231E"
BRAILLE_BASE = 0x2800
BRAILLE_DOT_BITS = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)
WORKING_ORBITS_PER_RADIUS = 0.55
WORKING_SAMPLES_PER_RADIUS = 1.6
WORKING_MIN_ORBITS = 14
WORKING_MIN_SAMPLES = 52
WORKING_PARTICLES_PER_ORBIT = 3
SOLVING_BACKGROUND_DENSITY = 0.11
SOLVING_SIGNAL_COUNT = 9
SOLVING_SIGNAL_TRAIL_STEPS = 3
SOLVING_SIGNAL_TRAIL_GAP = 0.06
EXTRACT_BRANCH_COUNT = 7
EXTRACT_TRAIL_STEPS = 3
EXTRACT_TRAIL_GAP = 0.055
SHARED_EDGE_INNER_RADIUS = 0.72
SHARED_EDGE_DENSITY = 0.29
STAGE_INTERIOR_RADIUS = 0.69
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
SUPERNOVA_CORE_START = 0.31
SUPERNOVA_REFORM_START = 0.66
SUPERNOVA_CORE_RADIUS = 0.1
SUPERNOVA_REFORM_END = 0.94


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
        caption="working...",
        accent=EVEROS_YELLOW,
    ),
    "ingesting": SphereState(
        key="ingesting",
        caption="capturing conversation into memory",
        accent=EVEROS_CYAN,
    ),
    "extracting": SphereState(
        key="extracting",
        caption="extracting episode -> atomic facts",
        accent=EVEROS_ORANGE,
    ),
    "indexing": SphereState(
        key="indexing",
        caption="organizing memory for fast recall",
        accent=EVEROS_CYAN,
    ),
    "recalling": SphereState(
        key="recalling",
        caption="scanning memory sphere",
        accent=EVEROS_GREEN,
    ),
    "remembered": SphereState(
        key="remembered",
        caption="found the matching memory",
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


def _build_solving_network(
    *, width: int, height: int, phase: float, state: SphereState
) -> DotSphereFrame:
    """Render Extract as a dense memory web with packets following its edges."""

    sub_width, sub_height, center_x, center_y, radius_x, radius_y = _sphere_geometry(
        width, height
    )
    animation_time = phase * math.tau
    yaw = animation_time * 0.12
    tilt = 0.32
    sin_tilt, cos_tilt = math.sin(tilt), math.cos(tilt)

    def project_at_yaw(
        x3: float,
        y3: float,
        z3: float,
        sample_yaw: float,
    ) -> tuple[int, int, float]:
        sample_sin_yaw = math.sin(sample_yaw)
        sample_cos_yaw = math.cos(sample_yaw)
        x_rotated = x3 * sample_cos_yaw + z3 * sample_sin_yaw
        z_rotated = -x3 * sample_sin_yaw + z3 * sample_cos_yaw
        y_projected = y3 * cos_tilt - z_rotated * sin_tilt
        depth = y3 * sin_tilt + z_rotated * cos_tilt
        return (
            round(center_x + x_rotated * radius_x * STAGE_INTERIOR_RADIUS),
            round(center_y - y_projected * radius_y * STAGE_INTERIOR_RADIUS),
            depth,
        )

    def project(x3: float, y3: float, z3: float) -> tuple[int, int, float]:
        return project_at_yaw(x3, y3, z3, yaw)

    surface_area = math.pi * radius_x * radius_y
    background_count = max(
        140,
        round(surface_area * SOLVING_BACKGROUND_DENSITY * STAGE_INTERIOR_RADIUS**2),
    )
    node_count = max(28, round(radius_x * 1.05))
    masks: dict[tuple[int, int], int] = {}
    depths: dict[tuple[int, int], float] = {}
    background_depths: dict[tuple[int, int], float] = {}
    signal_depths: dict[tuple[int, int], float] = {}
    node_depths: dict[tuple[int, int], float] = {}
    edge_depths: dict[tuple[int, int], float] = {}
    edge_visibilities: dict[tuple[int, int], float] = {}

    # A dense spherical field preserves the particle density of the other
    # stages. Each sample follows a small surface flow rather than remaining
    # fixed, while the zero-mean motion keeps the sphere centered.
    for index in range(background_count):
        base_y = 1 - 2 * ((index + 0.5) / background_count)
        base_latitude = math.asin(base_y)
        flow_speed = 0.12 + 0.08 * _stable_hash(index, 4.3)
        latitude = base_latitude + 0.04 * math.sin(
            animation_time * (0.4 + 0.15 * _stable_hash(index, 8.1))
            + index * GOLDEN_ANGLE * 0.37
        )
        latitude = max(-math.pi / 2, min(math.pi / 2, latitude))
        y3 = math.sin(latitude)
        latitude_radius = math.cos(latitude)
        theta = (
            index * GOLDEN_ANGLE
            + animation_time * flow_speed
            + 0.025 * math.sin(animation_time * 0.55 + index * 0.19)
        )
        x3 = latitude_radius * math.cos(theta)
        z3 = latitude_radius * math.sin(theta)
        sub_x, sub_y, depth = project(x3, y3, z3)
        if 0 <= sub_x < sub_width and 0 <= sub_y < sub_height:
            position = (sub_x // 2, sub_y // 4)
            _add_braille_dot(
                masks=masks,
                depths=depths,
                sub_x=sub_x,
                sub_y=sub_y,
                z=depth,
            )
            background_depths[position] = max(
                depth,
                background_depths.get(position, -1.0),
            )

    base_nodes: list[tuple[float, float, float]] = []
    nodes: list[tuple[float, float, float]] = []
    for index in range(node_count):
        base_y = 1 - 2 * ((index + 0.5) / node_count)
        latitude_radius = math.sqrt(max(0.0, 1.0 - base_y * base_y))
        theta = index * GOLDEN_ANGLE
        base_x = latitude_radius * math.cos(theta)
        base_z = latitude_radius * math.sin(theta)
        base_nodes.append((base_x, base_y, base_z))
        x3 = base_x + 0.12 * math.sin(animation_time * 0.72 + index * 0.31 + 9)
        y3 = base_y + 0.12 * math.sin(animation_time * 0.63 + index * 0.53 + 27)
        z3 = base_z + 0.12 * math.sin(animation_time * 0.81 + index * 0.77 + 55)
        nodes.append(_normalize_3d(x3, y3, z3))

    projected_nodes = [project(*node) for node in nodes]
    edge_threshold = 0.72
    edges: list[tuple[int, int, float]] = []
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for start_index in range(node_count):
        for end_index in range(start_index + 1, node_count):
            distance = math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(
                        base_nodes[start_index],
                        base_nodes[end_index],
                        strict=True,
                    )
                )
            )
            if distance < edge_threshold:
                edges.append((start_index, end_index, distance))
                adjacency[start_index].append(end_index)
                adjacency[end_index].append(start_index)

    for start_index, end_index, distance in edges:
        start_x, start_y, start_z = projected_nodes[start_index]
        end_x, end_y, end_z = projected_nodes[end_index]
        line_depth = (start_z + end_z) / 2
        depth_factor = 0.3 + 0.55 * ((line_depth + 1) / 2)
        visibility = (1 - distance / edge_threshold) * depth_factor
        steps = max(1, max(abs(end_x - start_x), abs(end_y - start_y)))
        for step in range(0, steps + 1, 2):
            progress = step / steps
            sub_x = round(start_x + (end_x - start_x) * progress)
            sub_y = round(start_y + (end_y - start_y) * progress)
            edge_depth = start_z + (end_z - start_z) * progress
            _add_braille_dot(
                masks=masks,
                depths=depths,
                sub_x=sub_x,
                sub_y=sub_y,
                z=edge_depth,
            )
            position = (sub_x // 2, sub_y // 4)
            edge_depths[position] = max(
                edge_depth,
                edge_depths.get(position, -1.0),
            )
            edge_visibilities[position] = max(
                visibility,
                edge_visibilities.get(position, 0.0),
            )

    for sub_x, sub_y, depth in projected_nodes:
        offsets = _particle_offsets_for_depth(depth)
        for offset_x, offset_y in offsets:
            node_x = sub_x + offset_x
            node_y = sub_y + offset_y
            if not _inside_sphere_projection(
                node_x,
                node_y,
                center_x,
                center_y,
                radius_x,
                radius_y,
            ):
                continue
            _add_braille_dot(
                masks=masks,
                depths=depths,
                sub_x=node_x,
                sub_y=node_y,
                z=depth,
            )
            position = (node_x // 2, node_y // 4)
            node_depths[position] = max(
                depth,
                node_depths.get(position, -1.0),
            )

    # Each bright packet follows one continuous walk through the connected
    # graph. Reaching a node therefore leads into the next edge instead of
    # respawning elsewhere, matching the reference's surface traversal.
    for signal in range(SOLVING_SIGNAL_COUNT):
        head_clock = animation_time * 0.46 + signal / SOLVING_SIGNAL_COUNT
        seed = round((signal + 0.5) * node_count / SOLVING_SIGNAL_COUNT) % node_count
        for trail_step in range(SOLVING_SIGNAL_TRAIL_STEPS):
            signal_clock = max(
                0.0,
                head_clock - trail_step * SOLVING_SIGNAL_TRAIL_GAP,
            )
            segment = math.floor(signal_clock)
            route = _signal_route_edge(
                adjacency=adjacency,
                seed=seed,
                segment=segment,
                signal=signal,
            )
            if route is None:
                continue
            start_index, end_index = route
            progress = signal_clock - math.floor(signal_clock)
            start_x, start_y, start_z = projected_nodes[start_index]
            end_x, end_y, end_z = projected_nodes[end_index]
            sub_x = round(start_x + (end_x - start_x) * progress)
            sub_y = round(start_y + (end_y - start_y) * progress)
            depth = start_z + (end_z - start_z) * progress
            for offset_x, offset_y in _particle_offsets_for_depth(
                depth,
                pulse=trail_step == 0,
            ):
                signal_x = sub_x + offset_x
                signal_y = sub_y + offset_y
                if not _inside_sphere_projection(
                    signal_x,
                    signal_y,
                    center_x,
                    center_y,
                    radius_x,
                    radius_y,
                ):
                    continue
                _add_braille_dot(
                    masks=masks,
                    depths=depths,
                    sub_x=signal_x,
                    sub_y=signal_y,
                    z=depth,
                )
                position = (signal_x // 2, signal_y // 4)
                signal_depths[position] = max(
                    depth,
                    signal_depths.get(position, -1.0),
                )

    shared_edge_positions = _replace_with_shared_outer_shell(
        masks=masks,
        depths=depths,
        layer_maps=(
            background_depths,
            signal_depths,
            node_depths,
            edge_depths,
            edge_visibilities,
        ),
        animation_time=animation_time,
        center_x=center_x,
        center_y=center_y,
        radius_x=radius_x,
        radius_y=radius_y,
    )

    cells = []
    for (x, y), mask in masks.items():
        signal_depth = signal_depths.get((x, y))
        node_depth = node_depths.get((x, y))
        edge_depth = edge_depths.get((x, y))
        background_depth = background_depths.get((x, y), -1.0)
        if (x, y) in shared_edge_positions:
            style = _style_for_shared_outer_shell(depths[(x, y)])
        elif signal_depth is not None:
            style = _style_for_network_signal(signal_depth)
        elif node_depth is not None and node_depth >= background_depth - 0.05:
            style = _style_for_network_node(node_depth)
        elif edge_depth is not None and edge_depth >= background_depth - 0.05:
            style = _style_for_network_edge(
                edge_depth,
                edge_visibilities[(x, y)],
            )
        else:
            style = _style_for_network_surface(background_depth)
        highlighted = signal_depth is not None
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


def _build_soft_supernova(
    *,
    width: int,
    height: int,
    phase: float,
    state: SphereState,
    progress: float,
) -> DotSphereFrame:
    """Burst the recalled sphere into a bright Braille-particle supernova."""

    progress = max(0.0, min(1.0, progress))
    if progress < SUPERNOVA_CORE_START:
        # Freeze the recalled source only while it explodes, so particle
        # identities cannot change in mid-flight.
        source = _build_working_cloud(
            width=width,
            height=height,
            phase=phase,
            state=SPHERE_STATES["recalling"],
        )
    else:
        # During the blank/reappearance section, use the *moving* Working
        # field at the phase the next idle frame will inherit. The last
        # celebration frame therefore joins the next loop without a jump.
        source = _build_working_cloud(
            width=width,
            height=height,
            phase=phase + progress * 2.4,
            state=SPHERE_STATES["ingesting"],
        )
    sub_width, sub_height, center_x, center_y, radius_x, radius_y = _sphere_geometry(
        width,
        height,
    )
    animation_time = progress * math.tau * 2.4

    contraction = _smoothstep(min(1.0, progress / 0.035))
    contraction_release = 1 - _smoothstep(
        max(0.0, min(1.0, (progress - 0.035) / 0.035))
    )
    contraction *= contraction_release
    source_scale_x = 1 - 0.085 * contraction
    source_scale_y = 1 - 0.065 * contraction

    flash = max(0.0, 1 - abs(progress - 0.047) / 0.03)
    wave_progress = max(0.0, min(1.0, (progress - 0.035) / 0.14))
    wave_radius = 0.08 + 1.02 * (1 - (1 - wave_progress) ** 2)
    wave_strength = 1 - _smoothstep(max(0.0, (progress - 0.17) / 0.07))

    masks: dict[tuple[int, int], int] = {}
    depths: dict[tuple[int, int], float] = {}
    source_styles: dict[tuple[int, int], str] = {}
    highlighted_positions: set[tuple[int, int]] = set()

    def edge_visibility(sub_x: int, sub_y: int, particle_id: int) -> float:
        """Feather particles before the rectangular terminal crop is visible."""

        if sub_x // 2 in {0, width - 1} or sub_y // 4 in {0, height - 1}:
            return 0.0
        distance = min(
            sub_x + 0.5,
            sub_width - 0.5 - sub_x,
            sub_y + 0.5,
            sub_height - 0.5 - sub_y,
        )
        feather_width = 2.5 + 5.5 * _stable_hash(particle_id, 17.3)
        return _smoothstep(max(0.0, min(1.0, distance / feather_width)))

    def add_particle(
        *,
        sub_x: int,
        sub_y: int,
        depth: float,
        style: str,
        highlighted: bool,
    ) -> None:
        if not (0 <= sub_x < sub_width and 0 <= sub_y < sub_height):
            return
        position = (sub_x // 2, sub_y // 4)
        if position not in depths or depth >= depths[position]:
            source_styles[position] = style
            if highlighted:
                highlighted_positions.add(position)
            else:
                highlighted_positions.discard(position)
        _add_braille_dot(
            masks=masks,
            depths=depths,
            sub_x=sub_x,
            sub_y=sub_y,
            z=depth,
        )

    for cell in source.cells:
        mask = ord(cell.glyph) - BRAILLE_BASE
        for local_x in range(2):
            for local_y in range(4):
                if not mask & BRAILLE_DOT_BITS[local_x][local_y]:
                    continue
                original_x = cell.x * 2 + local_x
                original_y = cell.y * 4 + local_y
                delta_x = original_x - center_x
                delta_y = original_y - center_y
                radial = math.sqrt(
                    (delta_x / radius_x) ** 2 + (delta_y / radius_y) ** 2
                )
                particle_id = original_y * sub_width + original_x
                if progress >= SUPERNOVA_CORE_START:
                    if progress < SUPERNOVA_REFORM_START:
                        core_growth = _smoothstep(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    (progress - SUPERNOVA_CORE_START) / 0.11,
                                ),
                            )
                        )
                        reveal_radius = SUPERNOVA_CORE_RADIUS * core_growth
                    else:
                        expansion = _smoothstep(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    (progress - SUPERNOVA_REFORM_START)
                                    / (SUPERNOVA_REFORM_END - SUPERNOVA_REFORM_START),
                                ),
                            )
                        )
                        reveal_radius = SUPERNOVA_CORE_RADIUS + expansion
                    # A soft radial edge makes the same moving ingest field
                    # appear first at its origin, hold there, and then expand
                    # continuously to the full sphere. No separate seed layer
                    # is swapped out when the rest of the particles arrive.
                    reveal_feather = 0.04
                    appearance = _smoothstep(
                        max(
                            0.0,
                            min(
                                1.0,
                                (reveal_radius - radial + reveal_feather)
                                / reveal_feather,
                            ),
                        )
                    )
                    if appearance <= 0.12:
                        continue
                    style = _blend_hex_color(
                        EVEROS_FIELD_BACKGROUND,
                        cell.style,
                        appearance,
                    )
                    add_particle(
                        sub_x=original_x,
                        sub_y=original_y,
                        depth=cell.z,
                        style=style,
                        highlighted=cell.highlighted and appearance > 0.45,
                    )
                    continue

                speed_hash = _stable_hash(particle_id, 44.1)
                spark_hash = _stable_hash(particle_id, 19.7)
                source_x = center_x + delta_x * source_scale_x
                source_y = center_y + delta_y * source_scale_y
                launch_start = 0.025 + 0.025 * speed_hash
                launch_duration = 0.03 + 0.055 * _stable_hash(
                    particle_id,
                    71.4,
                )
                launch_raw = max(
                    0.0,
                    min(1.0, (progress - launch_start) / launch_duration),
                )
                launch = 1 - (1 - launch_raw) ** 3

                # Decouple the launch direction from the particle's original
                # place on the sphere. A radial correlation turns the burst
                # into a larger circular shell instead of a chaotic release.
                scatter_angle = math.tau * _stable_hash(particle_id, 31.2)
                scatter_distance = math.sqrt(_stable_hash(particle_id, 57.8))
                field_radius = min(sub_width, sub_height) * 0.62
                irregular_envelope = (
                    0.78
                    + 0.14 * math.sin(scatter_angle * 3 + 1.1)
                    + 0.1 * math.sin(scatter_angle * 7 + 2.3)
                    + 0.08 * (_stable_hash(particle_id, 68.4) - 0.5)
                )
                target_radius = field_radius * scatter_distance * irregular_envelope
                target_x = center_x + math.cos(scatter_angle) * target_radius
                target_y = center_y + math.sin(scatter_angle) * target_radius
                launch_end = launch_start + launch_duration
                coast_elapsed = max(0.0, progress - launch_end)
                coast = _smoothstep(min(1.0, coast_elapsed / 0.08))
                travel_x = target_x - source_x
                travel_y = target_y - source_y
                travel_length = max(1.0, math.hypot(travel_x, travel_y))
                bend = (
                    (_stable_hash(particle_id, 83.7) - 0.5)
                    * min(sub_width, sub_height)
                    * 0.28
                )
                curve = math.sin(math.pi * launch)
                curve_x = -travel_y / travel_length * bend * curve
                curve_y = travel_x / travel_length * bend * curve
                drift_strength = launch * coast
                drift_angle = (
                    scatter_angle
                    + (_stable_hash(particle_id, 26.3) - 0.5) * math.pi * 0.75
                )
                coast_speed = 0.2 + 0.26 * _stable_hash(particle_id, 38.9)
                ballistic_x = (
                    math.cos(drift_angle)
                    * min(sub_width, sub_height)
                    * coast_elapsed
                    * coast_speed
                    * drift_strength
                )
                ballistic_y = (
                    math.sin(drift_angle)
                    * min(sub_width, sub_height)
                    * coast_elapsed
                    * coast_speed
                    * drift_strength
                )
                turbulence_x = (
                    math.sin(animation_time * 0.22 + particle_id * 0.13)
                    * 2.1
                    * drift_strength
                )
                turbulence_y = (
                    math.cos(animation_time * 0.19 + particle_id * 0.17)
                    * 1.7
                    * drift_strength
                )
                sub_x = round(
                    source_x + travel_x * launch + curve_x + ballistic_x + turbulence_x
                )
                sub_y = round(
                    source_y + travel_y * launch + curve_y + ballistic_y + turbulence_y
                )

                fade_start = 0.14 + 0.04 * _stable_hash(
                    particle_id,
                    91.6,
                )
                fade_duration = 0.1 + 0.04 * speed_hash
                fade_raw = max(
                    0.0,
                    min(1.0, (progress - fade_start) / fade_duration),
                )
                visibility = 1 - _smoothstep(fade_raw)
                if not (0 <= sub_x < sub_width and 0 <= sub_y < sub_height):
                    continue
                visibility *= edge_visibility(sub_x, sub_y, particle_id)
                if visibility <= 0.12:
                    continue
                wave = (
                    max(0.0, 1 - abs(radial - wave_radius) / 0.09) * wave_strength
                    if progress > 0.045
                    else 0.0
                )
                twinkle = (
                    launch
                    * visibility
                    * max(
                        0.0,
                        math.sin(animation_time * 0.72 + particle_id * 0.41),
                    )
                )
                glow = min(0.92, flash * 0.82 + wave * 0.58 + twinkle * 0.22)
                glow_target = (
                    EVEROS_CYAN
                    if cell.highlighted or (flash > 0 and radial < 0.32)
                    else EVEROS_YELLOW_PALE
                )
                style = _blend_hex_color(cell.style, glow_target, glow)
                style = _blend_hex_color(
                    style,
                    EVEROS_FIELD_BACKGROUND,
                    1 - visibility,
                )
                add_particle(
                    sub_x=sub_x,
                    sub_y=sub_y,
                    depth=cell.z,
                    style=style,
                    highlighted=cell.highlighted and visibility > 0.45,
                )

                trails_visible = (
                    math.sin(math.pi * launch_raw) * visibility
                    if 0 < launch_raw < 1
                    else 0.0
                )
                if spark_hash <= 0.54 or trails_visible <= 1e-6:
                    continue
                if launch_duration < 0.09 and spark_hash > 0.72:
                    trail_count = 3
                elif spark_hash > 0.76:
                    trail_count = 2
                else:
                    trail_count = 1
                tangent_x = travel_x + (
                    -travel_y
                    / travel_length
                    * bend
                    * math.pi
                    * math.cos(math.pi * launch)
                )
                tangent_y = travel_y + (
                    travel_x
                    / travel_length
                    * bend
                    * math.pi
                    * math.cos(math.pi * launch)
                )
                travel_angle = math.atan2(tangent_y, tangent_x)
                for trail_step in range(1, trail_count + 1):
                    distance = trail_step * (1.0 + trails_visible * 2.35)
                    trail_x = round(sub_x - math.cos(travel_angle) * distance)
                    trail_y = round(sub_y - math.sin(travel_angle) * distance)
                    if not (0 <= trail_x < sub_width and 0 <= trail_y < sub_height):
                        continue
                    trail_edge_visibility = edge_visibility(
                        trail_x,
                        trail_y,
                        particle_id,
                    )
                    if trail_edge_visibility <= 0.12:
                        continue
                    trail_style = _blend_hex_color(
                        style,
                        EVEROS_FIELD_BACKGROUND,
                        min(
                            0.94,
                            0.22 + trail_step * 0.18 + (1 - trails_visible) * 0.46,
                        ),
                    )
                    trail_style = _blend_hex_color(
                        trail_style,
                        EVEROS_FIELD_BACKGROUND,
                        1 - trail_edge_visibility,
                    )
                    add_particle(
                        sub_x=trail_x,
                        sub_y=trail_y,
                        depth=cell.z - trail_step * 0.02,
                        style=trail_style,
                        highlighted=False,
                    )

    cells = tuple(
        DotCell(
            x=x,
            y=y,
            z=depths[(x, y)],
            glyph=chr(BRAILLE_BASE + mask),
            style=source_styles[(x, y)],
            highlighted=(x, y) in highlighted_positions,
        )
        for (x, y), mask in sorted(
            masks.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    )
    return DotSphereFrame(
        width=width,
        height=height,
        state=state,
        cells=cells,
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


def _blend_hex_color(start: str, end: str, progress: float) -> str:
    """Blend the plain RGB styles used by the particle renderer."""

    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    channels = tuple(
        round(a + (b - a) * progress) for a, b in zip(start_rgb, end_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02X}" for channel in channels)


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


def _sphere_geometry(
    width: int,
    height: int,
) -> tuple[int, int, float, float, float, float]:
    """Return a physically round Braille projection for the available space."""

    sub_width = width * 2
    sub_height = height * 4
    center_x = (sub_width - 1) / 2
    center_y = (sub_height - 1) / 2 + 1
    radius_x = max(1.0, (center_x - 6) * 0.9)
    radius_y = max(1.0, (center_y - 5) * 0.9)
    return sub_width, sub_height, center_x, center_y, radius_x, radius_y


def _inside_sphere_projection(
    sub_x: int,
    sub_y: int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> bool:
    normalized = ((sub_x - center_x) / radius_x) ** 2 + (
        (sub_y - center_y) / radius_y
    ) ** 2
    return normalized <= 1.0


def _replace_with_shared_outer_shell(
    *,
    masks: dict[tuple[int, int], int],
    depths: dict[tuple[int, int], float],
    layer_maps: tuple[dict[tuple[int, int], float], ...],
    animation_time: float,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> set[tuple[int, int]]:
    """Give every processing state one identical, stable particle edge.

    Stage renderers are free to animate the center of the sphere. The outer
    band is replaced after that work so a change in network or orbit density
    cannot make the silhouette appear to jump between stages.
    """

    shared_masks: dict[tuple[int, int], int] = {}
    shared_depths: dict[tuple[int, int], float] = {}
    surface_area = math.pi * radius_x * radius_y
    band_ratio = 1 - SHARED_EDGE_INNER_RADIUS**2
    particle_count = max(
        120,
        round(surface_area * band_ratio * SHARED_EDGE_DENSITY),
    )

    for index in range(particle_count):
        angle = (
            index * GOLDEN_ANGLE
            + animation_time * 0.055
            + 0.028
            * math.sin(
                animation_time * (0.16 + 0.05 * _stable_hash(index, 29.4))
                + index * 0.43
            )
        )
        radial_hash = _stable_hash(index, 41.7)
        base_radius = math.sqrt(SHARED_EDGE_INNER_RADIUS**2 + band_ratio * radial_hash)
        radius = base_radius + 0.014 * math.sin(
            animation_time * (0.22 + 0.08 * _stable_hash(index, 63.1)) + index * 0.37
        )
        radius = max(
            SHARED_EDGE_INNER_RADIUS + 0.01,
            min(0.985, radius),
        )
        sub_x = round(center_x + math.cos(angle) * radius_x * radius)
        sub_y = round(center_y - math.sin(angle) * radius_y * radius)
        hemisphere = 1.0 if _stable_hash(index, 17.9) >= 0.38 else -1.0
        depth = hemisphere * math.sqrt(max(0.0, 1 - radius * radius))
        _add_braille_dot(
            masks=shared_masks,
            depths=shared_depths,
            sub_x=sub_x,
            sub_y=sub_y,
            z=depth,
        )

    # Symmetric pairs drift around each cardinal direction. Their small
    # tangential motion keeps the silhouette alive while preserving its exact
    # adaptive width and height through the whole animation cycle.
    boundary_wobble = 0.018 + 0.012 * (0.5 + 0.5 * math.sin(animation_time * 0.34))
    for cardinal in range(4):
        cardinal_angle = cardinal * math.pi / 2
        for direction in (-1, 1):
            angle = cardinal_angle + direction * boundary_wobble
            sub_x = round(center_x + math.cos(angle) * radius_x * 0.998)
            sub_y = round(center_y - math.sin(angle) * radius_y * 0.998)
            _add_braille_dot(
                masks=shared_masks,
                depths=shared_depths,
                sub_x=sub_x,
                sub_y=sub_y,
                z=0.0,
            )

    # Braille cells are larger than their sub-dots. Keep only cells whose
    # visual center belongs to the outer band so the moving shell never masks
    # a stage-specific packet travelling through the middle.
    for position in tuple(shared_masks):
        if (
            _cell_projection_radius(
                position,
                center_x=center_x,
                center_y=center_y,
                radius_x=radius_x,
                radius_y=radius_y,
            )
            < SHARED_EDGE_INNER_RADIUS
        ):
            shared_masks.pop(position)
            shared_depths.pop(position)

    replace_positions = set(shared_masks)
    replace_positions.update(
        position
        for position in masks
        if _cell_projection_radius(
            position,
            center_x=center_x,
            center_y=center_y,
            radius_x=radius_x,
            radius_y=radius_y,
        )
        >= SHARED_EDGE_INNER_RADIUS
    )
    for position in replace_positions:
        masks.pop(position, None)
        depths.pop(position, None)
        for layer_map in layer_maps:
            layer_map.pop(position, None)

    masks.update(shared_masks)
    depths.update(shared_depths)
    return set(shared_masks)


def _cell_projection_radius(
    position: tuple[int, int],
    *,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> float:
    """Return a terminal cell's radial position in the Braille projection."""

    cell_x, cell_y = position
    sub_x = cell_x * 2 + 0.5
    sub_y = cell_y * 4 + 1.5
    return math.sqrt(
        ((sub_x - center_x) / radius_x) ** 2 + ((sub_y - center_y) / radius_y) ** 2
    )


def _rotate_around_axis(
    point: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[float, float, float]:
    """Rotate a particle around one stable flow axis using Rodrigues' formula."""

    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    cross = _cross_3d(axis, point)
    dot = sum(a * b for a, b in zip(axis, point, strict=True))
    return tuple(
        point[index] * cos_angle
        + cross[index] * sin_angle
        + axis[index] * dot * (1 - cos_angle)
        for index in range(3)
    )


def _point_on_orbit(
    basis_u: tuple[float, float, float],
    basis_v: tuple[float, float, float],
    radius: float,
    angle: float,
    yaw: float,
    tilt: float,
) -> tuple[float, float, float]:
    point = tuple(
        (basis_u[index] * math.cos(angle) + basis_v[index] * math.sin(angle)) * radius
        for index in range(3)
    )
    point = _rotate_around_axis(point, (0.0, 1.0, 0.0), yaw)
    return _rotate_around_axis(point, (1.0, 0.0, 0.0), tilt)


def _particle_offsets_for_depth(
    depth: float,
    *,
    pulse: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Make near particles physically larger, mirroring the Canvas reference."""

    offsets = [(0, 0)]
    if depth > -0.45:
        offsets.append((1, 0))
    if depth > 0.1:
        offsets.append((0, 1))
    if depth > 0.55:
        offsets.append((1, 1))
    if pulse and depth > 0.15:
        offsets.append((-1, 0))
    return tuple(offsets)


def _style_for_active_particle(depth: float, *, allow_white: bool) -> str:
    depth_ratio = (depth + 1) / 2
    if allow_white and depth_ratio > 0.5:
        return EVEROS_CYAN
    if depth_ratio > 0.86:
        return EVEROS_YELLOW_PALE
    if depth_ratio > 0.7:
        return EVEROS_YELLOW
    if depth_ratio > 0.52:
        return EVEROS_GOLD_LIGHT
    if depth_ratio > 0.34:
        return EVEROS_GOLD_WARM
    if depth_ratio > 0.16:
        return EVEROS_GOLD_MID
    return EVEROS_GOLD_DARK


def _style_for_ghost_depth(depth: float) -> str:
    """Approximate the reference ghost-path alpha using dark gold steps."""

    depth_ratio = (depth + 1) / 2
    if depth_ratio > 0.82:
        return EVEROS_GOLD_MID
    if depth_ratio > 0.55:
        return EVEROS_GOLD_DARK
    if depth_ratio > 0.28:
        return EVEROS_GOLD_DEEP
    return EVEROS_GOLD_SHADOW


def _style_for_shared_outer_shell(depth: float) -> str:
    """Keep edge contrast calm and identical while the center tells the story."""

    if depth > 0.48:
        return EVEROS_GOLD_MID
    if depth > 0.18:
        return EVEROS_GOLD_DARK
    if depth > -0.18:
        return EVEROS_GOLD_DEEP
    return EVEROS_GOLD_SHADOW


def _style_for_network_node(depth: float) -> str:
    if depth > 0.72:
        return EVEROS_YELLOW_PALE
    if depth > 0.42:
        return EVEROS_YELLOW
    if depth > 0.12:
        return EVEROS_GOLD_LIGHT
    if depth > -0.18:
        return EVEROS_GOLD_WARM
    if depth > -0.5:
        return EVEROS_GOLD_MID
    if depth > -0.78:
        return EVEROS_GOLD_DEEP
    return EVEROS_GOLD_SHADOW


def _style_for_network_surface(depth: float) -> str:
    """Give Extract a bright front hemisphere and a dim visible back."""

    if depth > 0.65:
        return EVEROS_YELLOW
    if depth > 0.3:
        return EVEROS_GOLD_LIGHT
    if depth > 0.0:
        return EVEROS_GOLD_WARM
    if depth > -0.35:
        return EVEROS_GOLD_MID
    if depth > -0.68:
        return EVEROS_GOLD_DEEP
    return EVEROS_GOLD_SHADOW


def _style_for_network_signal(depth: float) -> str:
    """Keep packets white across the front and side, dimming only at the back."""

    if depth > -0.32:
        return EVEROS_CYAN
    if depth > -0.62:
        return EVEROS_GOLD_LIGHT
    return EVEROS_GOLD_DARK


def _style_for_network_edge(depth: float, visibility: float) -> str:
    if depth < -0.62:
        return EVEROS_GOLD_SHADOW
    if depth < -0.28:
        return EVEROS_GOLD_DEEP

    depth_ratio = (depth + 1) / 2
    ink = visibility * (0.55 + 0.45 * depth_ratio)
    if depth > 0.42 and ink > 0.17:
        return EVEROS_YELLOW
    if ink > 0.27:
        return EVEROS_GOLD_LIGHT
    if ink > 0.17:
        return EVEROS_GOLD_WARM
    if ink > 0.09:
        return EVEROS_GOLD_MID
    if ink > 0.04:
        return EVEROS_GOLD_DARK
    return EVEROS_GOLD_DEEP


def _normalize_3d(x: float, y: float, z: float) -> tuple[float, float, float]:
    length = max(1e-6, math.sqrt(x * x + y * y + z * z))
    return x / length, y / length, z / length


def _cross_3d(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _signal_route_edge(
    *,
    adjacency: list[list[int]],
    seed: int,
    segment: int,
    signal: int,
) -> tuple[int, int] | None:
    """Return one edge in a deterministic, continuous graph walk."""

    if not adjacency or not adjacency[seed]:
        return None
    segment = max(0, segment)

    def next_node(previous: int, current: int) -> int:
        choices = [node for node in adjacency[current] if node != previous]
        if not choices:
            choices = adjacency[current]
        selector = _stable_hash(
            current * 131 + max(previous, 0) * 17,
            signal * 5.3 + 1.9,
        )
        return choices[min(len(choices) - 1, math.floor(selector * len(choices)))]

    # The next hop depends only on (previous, current), so the finite graph
    # eventually cycles. Detect that cycle to keep long-running demos cheap.
    states: list[tuple[int, int]] = []
    seen: dict[tuple[int, int], int] = {}
    state = (-1, seed)
    while state not in seen:
        seen[state] = len(states)
        states.append(state)
        previous, current = state
        state = (current, next_node(previous, current))

    if segment < len(states):
        state_index = segment
    else:
        cycle_start = seen[state]
        cycle_length = len(states) - cycle_start
        state_index = cycle_start + (segment - cycle_start) % cycle_length

    previous, current = states[state_index]
    return current, next_node(previous, current)


def _stable_hash(value: int, salt: float) -> float:
    """Return a deterministic pseudo-random value in [0, 1)."""

    hashed = math.sin((value + 1) * 12.9898 + salt * 78.233) * 43758.5453
    return hashed - math.floor(hashed)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def _style_for_depth(z: float, state: SphereState) -> str:
    if state.key == "extracting" and z > 0.38:
        return EVEROS_ORANGE
    if state.key == "indexing" and z > 0.3:
        return EVEROS_CYAN
    if state.key == "ingesting" and z > 0.68:
        return EVEROS_CYAN
    if z > 0.78:
        return EVEROS_YELLOW_PALE
    if z > 0.68:
        return EVEROS_YELLOW
    if z > 0.56:
        return EVEROS_GOLD_LIGHT
    if z > 0.44:
        return EVEROS_GOLD_WARM
    if z > 0.25:
        return EVEROS_GOLD_MID
    if z > 0:
        return EVEROS_GOLD_DARK
    if z > -0.4:
        return EVEROS_GOLD_DEEP
    return EVEROS_GOLD_SHADOW


def _state_local_phase(phase: float, state_key: str) -> float:
    state_keys = tuple(SPHERE_STATES)
    return (phase * len(state_keys) - state_keys.index(state_key)) % 1.0
