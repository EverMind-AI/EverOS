"""Render README media for the ``everos demo`` TUI."""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio

TERMINAL_SIZE = (150, 48)
FRAME_SECONDS = 0.85


@dataclass(frozen=True)
class FramePlan:
    state: str
    phase: float


@dataclass(frozen=True)
class OpacitySchedule:
    key_times: str
    values: str


def build_frame_plan(states: Sequence[str]) -> tuple[FramePlan, ...]:
    """Return animation frames, closed by a duplicate of the first frame."""
    if not states:
        raise ValueError("at least one animation state is required")

    state_count = len(states)
    frames = [
        FramePlan(state=state, phase=index / state_count)
        for index, state in enumerate(states)
    ]
    frames.append(FramePlan(state=states[0], phase=0.0))
    return tuple(frames)


def opacity_schedule(*, index: int, total: int) -> OpacitySchedule:
    """Build a gap-free discrete opacity schedule for one frame."""
    if total < 2:
        raise ValueError("at least two frames are required")
    if index < 0 or index >= total:
        raise ValueError("frame index is out of range")

    start = index / total
    end = (index + 1) / total

    if index == 0:
        return OpacitySchedule(key_times=f"0;{_time(end)};1", values="1;0;0")
    if index == total - 1:
        return OpacitySchedule(key_times=f"0;{_time(start)};1", values="0;1;1")
    return OpacitySchedule(
        key_times=f"0;{_time(start)};{_time(end)};1",
        values="0;1;0;0",
    )


async def render_media(out_dir: Path) -> tuple[Path, Path]:
    """Render the static screenshot and looping animated SVG."""
    from everos.entrypoints.cli.demo_tui import DotSphereWidget

    await anyio.Path(out_dir).mkdir(parents=True, exist_ok=True)
    screenshot = out_dir / "everos-demo-tui-screenshot.svg"
    remembered_index = DotSphereWidget.STATES.index("remembered")
    remembered = FramePlan(
        state=DotSphereWidget.STATES[remembered_index],
        phase=remembered_index / len(DotSphereWidget.STATES),
    )
    await _export_frame(screenshot, remembered)

    plan = build_frame_plan(DotSphereWidget.STATES)
    frame_paths: list[Path] = []
    for index, frame in enumerate(plan):
        frame_path = out_dir / f"frame-{index:02d}-{frame.state}.svg"
        await _export_frame(frame_path, frame)
        frame_paths.append(frame_path)

    animation = out_dir / "everos-demo-tui-animation.svg"
    animation.write_text(_build_animation_svg(frame_paths, plan))
    return screenshot, animation


async def _export_frame(path: Path, frame: FramePlan) -> None:
    from everos.entrypoints.cli.demo_sphere import (
        build_dot_sphere,
        render_dot_sphere_text,
    )
    from everos.entrypoints.cli.demo_tui import (
        SPHERE_FRAME_HEIGHT,
        SPHERE_FRAME_WIDTH,
        DotSphereWidget,
        EverOSDemoApp,
    )

    app = EverOSDemoApp()
    async with app.run_test(size=TERMINAL_SIZE) as pilot:
        await pilot.pause(0.05)
        widget = app.query_one(DotSphereWidget)
        widget.pause_animation()
        sphere = build_dot_sphere(
            width=SPHERE_FRAME_WIDTH,
            height=SPHERE_FRAME_HEIGHT,
            phase=frame.phase,
            state_key=frame.state,
        )
        widget.update(render_dot_sphere_text(sphere))
        await pilot.pause(0.05)
        app.save_screenshot(str(path))
    path.write_text(normalize_svg_terminal_ids(path.read_text()))


def _build_animation_svg(frame_paths: Sequence[Path], plan: Sequence[FramePlan]) -> str:
    if len(frame_paths) != len(plan):
        raise ValueError("frame paths and frame plan lengths must match")

    view_box, width, height = _read_svg_dimensions(frame_paths[0])
    duration = len(frame_paths) * FRAME_SECONDS
    images = []
    for index, frame_path in enumerate(frame_paths):
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        schedule = opacity_schedule(index=index, total=len(frame_paths))
        state = html.escape(plan[index].state)
        images.append(
            f'  <image width="{width}" height="{height}" '
            f'href="data:image/svg+xml;base64,{encoded}" opacity="0">\n'
            f"    <title>EverOS demo state: {state}</title>\n"
            f'    <animate attributeName="opacity" dur="{duration:.2f}s" '
            f'repeatCount="indefinite" keyTimes="{schedule.key_times}" '
            f'values="{schedule.values}" calcMode="discrete" />\n'
            "  </image>"
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="{view_box}">\n'
        "<title>Animated EverOS demo terminal UI</title>\n"
        "<desc>EverOS demo cycles through memory states with a closed loop.</desc>\n"
        + "\n".join(images)
        + "\n</svg>\n"
    )


def _read_svg_dimensions(path: Path) -> tuple[str, str, str]:
    svg_open = re.search(r"<svg\s+([^>]+)>", path.read_text())
    if svg_open is None:
        raise ValueError(f"could not find SVG root in {path}")
    view_box_match = re.search(r'viewBox="([^"]+)"', svg_open.group(0))
    if view_box_match is None:
        raise ValueError(f"could not find SVG viewBox in {path}")
    view_box = view_box_match.group(1)
    _, _, width, height = view_box.split()
    return view_box, width, height


def normalize_svg_terminal_ids(svg: str) -> str:
    """Normalize Rich's random terminal SVG prefix for deterministic media."""
    return re.sub(r"terminal-\d+", "terminal-everos", svg)


def _time(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/everos-demo-media"),
        help="Directory for generated README SVG media.",
    )
    args = parser.parse_args()
    screenshot, animation = asyncio.run(render_media(args.out_dir))
    print(screenshot)
    print(animation)


if __name__ == "__main__":
    main()
