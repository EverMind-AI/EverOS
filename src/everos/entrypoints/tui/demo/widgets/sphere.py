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
SOLVING_BACKGROUND_DENSITY = 0.21
SOLVING_SIGNAL_COUNT = 9
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
    if state.key in {
        "booting",
        "ingesting",
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
    if state.key == "extracting":
        return _build_solving_network(
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
                if position not in highlighted_positions and depths[position] > -0.15
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
        active_depth = active_depths.get((x, y))
        target_highlighted = (x, y) in highlighted_positions
        if target_highlighted and state.key == "recalling":
            style = EVEROS_CYAN
        elif target_highlighted:
            style = EVEROS_YELLOW
        elif state.key == "indexing" and depths[(x, y)] > 0.3:
            # Preserve the old Index behavior: the organized front layer turns
            # white, now projected onto the complete orbital sphere.
            style = EVEROS_CYAN
        elif active_depth is not None and state.key == "ingesting":
            style = _style_for_active_particle(active_depth, allow_white=True)
        elif active_depth is not None:
            style = _style_for_active_particle(active_depth, allow_white=False)
        else:
            style = _style_for_ghost_depth(depths[(x, y)])
        highlighted = target_highlighted or style == EVEROS_CYAN
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
            round(center_x + x_rotated * radius_x),
            round(center_y - y_projected * radius_y),
            depth,
        )

    def project(x3: float, y3: float, z3: float) -> tuple[int, int, float]:
        return project_at_yaw(x3, y3, z3, yaw)

    surface_area = math.pi * radius_x * radius_y
    background_count = max(
        500,
        round(surface_area * SOLVING_BACKGROUND_DENSITY),
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

    # Cardinal surface samples keep the adaptive silhouette identical to the
    # orbital state without drawing a separate outline.
    for sub_x, sub_y in (
        (round(center_x - radius_x), round(center_y)),
        (round(center_x + radius_x), round(center_y)),
        (round(center_x), round(center_y - radius_y)),
        (round(center_x), round(center_y + radius_y)),
    ):
        position = (sub_x // 2, sub_y // 4)
        _add_braille_dot(
            masks=masks,
            depths=depths,
            sub_x=sub_x,
            sub_y=sub_y,
            z=0.0,
        )
        background_depths[position] = max(
            0.0,
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
        signal_clock = animation_time * 0.46 + signal / SOLVING_SIGNAL_COUNT
        segment = math.floor(signal_clock)
        seed = round((signal + 0.5) * node_count / SOLVING_SIGNAL_COUNT) % node_count
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
        for offset_x, offset_y in _particle_offsets_for_depth(depth, pulse=True):
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

    cells = []
    for (x, y), mask in masks.items():
        signal_depth = signal_depths.get((x, y))
        node_depth = node_depths.get((x, y))
        edge_depth = edge_depths.get((x, y))
        background_depth = background_depths.get((x, y), -1.0)
        if signal_depth is not None:
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
    if depth > 0.48 and ink > 0.22:
        return EVEROS_YELLOW
    if ink > 0.34:
        return EVEROS_GOLD_LIGHT
    if ink > 0.22:
        return EVEROS_GOLD_WARM
    if ink > 0.12:
        return EVEROS_GOLD_MID
    if ink > 0.06:
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
