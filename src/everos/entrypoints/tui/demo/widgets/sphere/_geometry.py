"""Projection, Braille sub-cell plotting and 3D helpers shared by renderers.

A terminal cell holds a 2x4 Braille dot grid, so renderers plot in sub-cell
coordinates and fold dots into cell masks here. Every renderer also shares one
outer particle shell so state changes never redraw the silhouette.
"""

from __future__ import annotations

import math

from everos.entrypoints.tui.demo.widgets.sphere._models import (
    BRAILLE_DOT_BITS,
    GOLDEN_ANGLE,
    SHARED_EDGE_DENSITY,
    SHARED_EDGE_INNER_RADIUS,
)


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


def _stable_hash(value: int, salt: float) -> float:
    """Return a deterministic pseudo-random value in [0, 1)."""

    hashed = math.sin((value + 1) * 12.9898 + salt * 78.233) * 43758.5453
    return hashed - math.floor(hashed)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)
