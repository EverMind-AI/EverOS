"""Golden frames pin the demo dot-sphere renderer byte for byte.

The sphere is pure geometry, so any refactor of its modules must leave every
frame identical. Each case is serialised (cells plus the rendered Rich markup),
hashed, and compared with ``fixtures/sphere_golden.json``. Depth is rounded to
1e-4 before hashing: libm trig differs in the last ulp between macOS and glibc,
which moves ``repr(z)`` but never a glyph, style or cell position. Regenerate
the fixture only for an intended visual change, with ``write_golden_fixture()``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from everos.entrypoints.tui.demo.widgets.sphere import (
    SPHERE_STATES,
    DotSphereFrame,
    blend_dot_sphere_frames,
    build_dot_sphere,
    render_dot_sphere_text,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sphere_golden.json"
SIZES = ((41, 19), (37, 17), (13, 7))
PHASES = (0.0, 0.2, 0.4, 0.6, 0.8)
SUPERNOVA_PROGRESS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
BLENDS = (
    ("ingesting", "extracting", 0.3),
    ("ingesting", "extracting", 0.7),
    ("recalling", "celebrating", 0.5),
)


def _serialise(frame: DotSphereFrame) -> str:
    cells = [
        [cell.x, cell.y, round(cell.z, 4), cell.glyph, cell.style, cell.highlighted]
        for cell in frame.cells
    ]
    payload = {
        "width": frame.width,
        "height": frame.height,
        "state": [frame.state.key, frame.state.caption, frame.state.accent],
        "cells": cells,
        "markup": render_dot_sphere_text(frame).markup,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _digest(frame: DotSphereFrame) -> str:
    return hashlib.sha256(_serialise(frame).encode("utf-8")).hexdigest()


def _golden_frames() -> Iterator[tuple[str, DotSphereFrame]]:
    for width, height in SIZES:
        for state_key in SPHERE_STATES:
            for phase in PHASES:
                frame = build_dot_sphere(
                    width=width, height=height, phase=phase, state_key=state_key
                )
                yield f"{width}x{height}/{state_key}/phase={phase}", frame
        for progress in SUPERNOVA_PROGRESS:
            frame = build_dot_sphere(
                width=width,
                height=height,
                phase=0.25,
                state_key="celebrating",
                state_phase=progress,
            )
            yield f"{width}x{height}/celebrating/state_phase={progress}", frame
        for previous_key, current_key, progress in BLENDS:
            previous = build_dot_sphere(
                width=width, height=height, phase=0.3, state_key=previous_key
            )
            current = build_dot_sphere(
                width=width, height=height, phase=0.35, state_key=current_key
            )
            frame = blend_dot_sphere_frames(previous, current, progress)
            name = f"{width}x{height}/blend/{previous_key}->{current_key}@{progress}"
            yield name, frame


def compute_golden_digests() -> dict[str, str]:
    """Return the sha256 digest of every golden case, keyed by case name."""
    return {name: _digest(frame) for name, frame in _golden_frames()}


def write_golden_fixture() -> None:
    """Rewrite the committed fixture from the current renderer."""
    digests = compute_golden_digests()
    FIXTURE_PATH.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def expected_digests() -> dict[str, str]:
    return json.loads(FIXTURE_PATH.read_text())


def test_golden_case_set_matches_fixture(expected_digests: dict[str, str]) -> None:
    assert sorted(compute_golden_digests()) == sorted(expected_digests)


def test_every_golden_frame_is_unchanged(expected_digests: dict[str, str]) -> None:
    changed = [
        name
        for name, digest in compute_golden_digests().items()
        if expected_digests.get(name) != digest
    ]
    assert changed == []
