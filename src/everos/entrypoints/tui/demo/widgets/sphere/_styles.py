"""Depth-to-colour rules for sphere particles, plus the colour blend.

Every renderer picks a particle's colour from its projected depth; keeping
those rules together means a re-shading touches one module. The palette itself
lives in :mod:`everos.entrypoints.tui.demo.theme`; styles stay plain
``#RRGGBB`` strings so frames can be blended channel-wise.
"""

from __future__ import annotations

from everos.entrypoints.tui.demo.theme import (
    EVEROS_CYAN,
    EVEROS_GOLD_DARK,
    EVEROS_GOLD_DEEP,
    EVEROS_GOLD_LIGHT,
    EVEROS_GOLD_MID,
    EVEROS_GOLD_SHADOW,
    EVEROS_GOLD_WARM,
    EVEROS_YELLOW,
    EVEROS_YELLOW_PALE,
)


def _blend_hex_color(start: str, end: str, progress: float) -> str:
    """Blend the plain RGB styles used by the particle renderer."""

    start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    channels = tuple(
        round(a + (b - a) * progress) for a, b in zip(start_rgb, end_rgb, strict=True)
    )
    return "#" + "".join(f"{channel:02X}" for channel in channels)


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
