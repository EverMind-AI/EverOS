"""EverOS demo README media generation contracts."""

from __future__ import annotations

import html

from scripts.render_demo_readme_media import (
    FRAME_SECONDS,
    FramePlan,
    _export_frame,
    build_frame_plan,
    normalize_svg_terminal_ids,
    opacity_schedule,
)


def test_readme_animation_frame_plan_closes_on_first_frame() -> None:
    plan = build_frame_plan(("booting", "ingesting", "source"), frames_per_state=3)

    assert plan[0] == plan[-1]
    assert len(plan) == 10
    assert [frame.state for frame in plan] == [
        "booting",
        "booting",
        "booting",
        "ingesting",
        "ingesting",
        "ingesting",
        "source",
        "source",
        "source",
        "booting",
    ]
    assert [frame.phase for frame in plan] == [
        0 / 9,
        1 / 9,
        2 / 9,
        3 / 9,
        4 / 9,
        5 / 9,
        6 / 9,
        7 / 9,
        8 / 9,
        0.0,
    ]


def test_readme_animation_default_plan_uses_smoother_frame_rate() -> None:
    states = ("booting", "ingesting", "source")

    plan = build_frame_plan(states)

    assert len(plan) >= len(states) * 7 + 1
    assert FRAME_SECONDS <= 0.10
    assert 1 / FRAME_SECONDS >= 10


def test_readme_animation_opacity_schedule_has_no_blank_gap() -> None:
    schedules = [opacity_schedule(index=idx, total=4) for idx in range(4)]

    assert schedules[0].values == "1;0;0"
    assert schedules[-1].values == "0;1;1"
    assert all("0.0005" not in schedule.key_times for schedule in schedules)


def test_readme_media_normalizes_rich_terminal_random_ids() -> None:
    svg = (
        "<style>.terminal-12345-matrix {}</style>"
        '<clipPath id="terminal-12345-line-0"></clipPath>'
        '<text class="terminal-12345-r1" '
        'clip-path="url(#terminal-12345-line-0)">EverOS</text>'
    )

    normalized = normalize_svg_terminal_ids(svg)

    assert "terminal-12345" not in normalized
    assert normalized.count("terminal-everos") == 4


async def test_export_frame_freezes_animation_before_rendering_requested_state(
    tmp_path, monkeypatch
) -> None:
    from everos.entrypoints.cli.demo_tui import DotSphereWidget

    def fast_mount(self: DotSphereWidget) -> None:
        self._animation_timer = self.set_interval(0.001, self._advance)
        self._advance()

    monkeypatch.setattr(DotSphereWidget, "on_mount", fast_mount)
    path = tmp_path / "frame.svg"

    await _export_frame(path, FramePlan(state="booting", phase=0.0))

    svg = html.unescape(path.read_text()).replace("\xa0", " ")
    assert "forming local memory field" in svg
    assert "ingesting conversation dots" not in svg
