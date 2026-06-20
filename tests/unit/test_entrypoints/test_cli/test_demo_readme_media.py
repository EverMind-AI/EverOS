"""EverOS demo README media generation contracts."""

from __future__ import annotations

from scripts.render_demo_readme_media import build_frame_plan, opacity_schedule


def test_readme_animation_frame_plan_closes_on_first_frame() -> None:
    plan = build_frame_plan(("booting", "ingesting", "source"))

    assert plan[0] == plan[-1]
    assert [frame.state for frame in plan] == [
        "booting",
        "ingesting",
        "source",
        "booting",
    ]


def test_readme_animation_opacity_schedule_has_no_blank_gap() -> None:
    schedules = [opacity_schedule(index=idx, total=4) for idx in range(4)]

    assert schedules[0].values == "1;0;0"
    assert schedules[-1].values == "0;1;1"
    assert all("0.0005" not in schedule.key_times for schedule in schedules)
