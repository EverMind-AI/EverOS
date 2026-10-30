"""The celebration renderer: the recalled sphere bursts and reforms.

The burst starts from a frozen recall frame so particle identities stay fixed
in flight; the reform uses the moving ingest field the next idle frame
inherits, so the idle loop wraps without a visible jump.
"""

from __future__ import annotations

import math

from everos.entrypoints.tui.demo.theme import (
    EVEROS_CYAN,
    EVEROS_FIELD_BACKGROUND,
    EVEROS_YELLOW_PALE,
)
from everos.entrypoints.tui.demo.widgets.sphere._geometry import (
    _add_braille_dot,
    _smoothstep,
    _sphere_geometry,
    _stable_hash,
)
from everos.entrypoints.tui.demo.widgets.sphere._models import (
    BRAILLE_BASE,
    BRAILLE_DOT_BITS,
    SPHERE_STATES,
    SUPERNOVA_CORE_RADIUS,
    SUPERNOVA_CORE_START,
    SUPERNOVA_REFORM_END,
    SUPERNOVA_REFORM_START,
    DotCell,
    DotSphereFrame,
    SphereState,
)
from everos.entrypoints.tui.demo.widgets.sphere._styles import _blend_hex_color
from everos.entrypoints.tui.demo.widgets.sphere._working import _build_working_cloud


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
