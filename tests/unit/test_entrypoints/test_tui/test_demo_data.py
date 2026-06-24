"""EverOS demo story data contracts."""

from __future__ import annotations

from everos.entrypoints.tui.demo.data import (
    DEFAULT_MEMORY_SEED,
    DEFAULT_QUERY,
    DemoStory,
    default_demo_story,
)


def test_default_demo_story_is_the_static_showcase() -> None:
    story = default_demo_story()

    assert isinstance(story, DemoStory)
    assert story.memory == DEFAULT_MEMORY_SEED
    assert story.answer == "Yosemite every spring"
    assert story.source_filename == "episode-2026-06-20.md"


def test_default_query_constant_is_available() -> None:
    assert DEFAULT_QUERY
