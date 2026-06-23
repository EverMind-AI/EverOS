"""EverOS playable demo story contracts."""

from __future__ import annotations

from everos.entrypoints.tui.demo.data import (
    DEFAULT_MEMORY_SEED,
    DEFAULT_QUERY,
    build_demo_story,
)


def test_demo_story_preserves_prompted_memory_and_query() -> None:
    story = build_demo_story(
        "I keep my Monday design review notes in Notion.",
        "Where are my Monday review notes?",
    )

    assert story.owner == "you"
    assert story.memory == "I keep my Monday design review notes in Notion."
    assert story.query == "Where are my Monday review notes?"
    assert story.answer == "I keep my Monday design review notes in Notion."
    assert story.source_filename == "episode-demo.md"
    assert story.fact_filename == "atomic_fact-demo.md"


def test_demo_story_keeps_default_yosemite_success_moment() -> None:
    story = build_demo_story(DEFAULT_MEMORY_SEED, DEFAULT_QUERY)

    assert story.answer == "Yosemite every spring"
