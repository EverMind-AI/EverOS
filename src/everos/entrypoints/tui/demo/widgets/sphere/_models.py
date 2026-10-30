"""Frame data types, the named sphere states and the renderer tuning knobs.

A frame is an immutable value (``DotSphereFrame`` of ``DotCell``) so renderers
stay pure functions and frames can be compared, blended and hashed in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from everos.entrypoints.tui.demo.theme import (
    EVEROS_CYAN,
    EVEROS_GREEN,
    EVEROS_ORANGE,
    EVEROS_YELLOW,
    EVEROS_YELLOW_SOFT,
)

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
EXTRACT_BRANCH_COUNT = 7
EXTRACT_TRAIL_STEPS = 3
EXTRACT_TRAIL_GAP = 0.055
SHARED_EDGE_INNER_RADIUS = 0.72
SHARED_EDGE_DENSITY = 0.29
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


def _state_local_phase(phase: float, state_key: str) -> float:
    state_keys = tuple(SPHERE_STATES)
    return (phase * len(state_keys) - state_keys.index(state_key)) % 1.0
